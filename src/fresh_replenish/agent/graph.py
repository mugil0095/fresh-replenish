"""The replenishment agent (spec section 11): one LangGraph agent with four plain-code tools.

agent -> tools -> limit check (a human takes over after 3 failed checks) -> agent
      -> human approval for each order that failed its limits -> number check -> end.
"""

import json
import uuid
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command, interrupt

from fresh_replenish.agent.checker import unsupported
from fresh_replenish.agent.tools import Toolbox

SYSTEM = """You help dark-store managers decide daily orders of fresh food.
- Every number you state must come from a tool output: quote it as given, never compute one.
- For an order, call compute_order and explain it with the get_forecast range and the cost ratio
  (lost-sale cost / waste cost; 9 by default, which orders at the 90th percentile).
- The forecast range runs from the 10th to the 90th percentile; the median is the 50th.
- What-if answers must say they are model-based estimates, not proof of cause and effect.
- If a tool returns an error, say the data is not available; never guess.
- If an order is outside the limits, say so: a planner must approve it.
- When asked to place or record an order, call write_decision_record after compute_order;
  never call it otherwise.
- Politely refuse anything that is not about ordering fresh food for these stores.
Dates are YYYY-MM-DD. A discount is the price as a share of normal (0.8 means 20% off)."""
REFUSAL = (
    "I can't back every number in my answer with a tool output, so I won't guess. "
    "Please ask a planner."
)
MAX_RETRIES = 2  # answers the number checker may reject before the agent gives up
MAX_FAILED_CHECKS = 3  # failed limit checks since the last review before a human takes over


class State(MessagesState):
    retries: int
    approval: str  # the latest planner decision: "approved" or "rejected"
    reviewed: int  # compute_order results already put to a planner
    approved: list  # [store_id, product_id, date, order] lines a planner approved


def _payload(message: ToolMessage):
    try:
        return json.loads(message.content)
    except (TypeError, ValueError):
        return message.content


def _orders(state: State) -> list[dict]:
    """Every compute_order result in the conversation that ran its limit check."""
    results = [
        _payload(m) for m in state["messages"]
        if isinstance(m, ToolMessage) and m.name == "compute_order"
    ]
    return [r for r in results if isinstance(r, dict) and "within_limits" in r]


def _failed(state: State) -> list[int]:
    """Positions of the compute_order results since the last review that failed their limits."""
    orders, start = _orders(state), state.get("reviewed", 0)
    return [i for i in range(start, len(orders)) if not orders[i]["within_limits"]]


def sources(messages: list) -> list[str]:
    """What an answer's numbers may come from: tool outputs and the user's (or planner's) words.

    Not the rules: their 9, 50 or 90 must not pass as an order quantity.
    """
    return [
        m.text for m in messages
        if isinstance(m, ToolMessage) or (isinstance(m, HumanMessage) and m.name != "checker")
    ]


def _key(store_id, product_id, date, order) -> list:
    return [int(store_id), int(product_id), str(date), round(float(order), 2)]


def make_tools(toolbox: Toolbox) -> list[StructuredTool]:
    def write_decision_record(
        store_id: int, product_id: int, date: str, order: float, reason: str,
        state: Annotated[dict, InjectedState],
    ) -> dict:
        """Save the order, its reason and this conversation's tool outputs."""
        # Checked here, not from earlier messages: a call in the same step can't see them.
        with toolbox.lock:  # parallel calls run in threads: check and save as one step
            check = toolbox.compute_order(store_id, product_id, date, units=order)
            if "error" in check:
                return check
            approved = _key(store_id, product_id, date, order) in state.get("approved", [])
            if not check["within_limits"] and not approved:
                return {"error": "this order is outside the limits: a planner must approve it"}
            outputs = [_payload(m) for m in state["messages"] if isinstance(m, ToolMessage)]
            return toolbox.write_decision_record(store_id, product_id, date, order, reason, outputs)

    methods = (toolbox.get_forecast, toolbox.compute_order, toolbox.what_if)
    return [StructuredTool.from_function(f) for f in (*methods, write_decision_record)]


def build_agent(model, toolbox: Toolbox):
    tools = make_tools(toolbox)
    bound = model.bind_tools(tools)

    def agent(state: State) -> dict:
        return {"messages": [bound.invoke([SystemMessage(SYSTEM), *state["messages"]])]}

    def after_agent(state: State) -> str:
        if state["messages"][-1].tool_calls:
            return "tools"
        return "approval" if _failed(state) else "check"

    def limits(state: State) -> Command:
        return Command(goto="approval" if len(_failed(state)) >= MAX_FAILED_CHECKS else "agent")

    def approval(state: State) -> dict:
        orders, failed = _orders(state), _failed(state)
        # After 3 failed checks a planner takes over all of them; otherwise each one in turn.
        index = failed[-1] if len(failed) >= MAX_FAILED_CHECKS else failed[0]
        line = orders[index]
        decision = interrupt({"order": line, "failed_checks": len(failed)})
        verdict = "approved" if decision.get("approved") else "rejected"
        approved = state.get("approved", []) + (
            [_key(line["store_id"], line["product_id"], line["date"], line["order"])]
            if verdict == "approved" else []
        )
        order = (f"{line['order']} units of product {line['product_id']} for store "
                 f"{line['store_id']} on {line['date']}")
        note = HumanMessage(f"A planner {verdict} the order of {order}. "
                            f"{decision.get('note', '')}".strip())
        return {
            "approval": verdict, "reviewed": index + 1, "approved": approved,
            "messages": [note],
        }

    def check(state: State) -> Command:
        bad = unsupported(state["messages"][-1].text, sources(state["messages"]))
        if not bad:
            return Command(goto=END)
        retries = state.get("retries", 0)
        if retries >= MAX_RETRIES:
            saved = any(
                m.name == "write_decision_record" and _payload(m) == {"saved": True}
                for m in state["messages"] if isinstance(m, ToolMessage)
            )
            note = " An order from this conversation was already saved." if saved else ""
            return Command(goto=END, update={"messages": [AIMessage(REFUSAL + note)]})
        retry = HumanMessage(
            f"The number checker rejected {', '.join(bad)}: these are not in any tool output. "
            "Answer again using only numbers from the tool outputs.", name="checker",
        )
        return Command(goto="agent", update={"retries": retries + 1, "messages": [retry]})

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("limits", limits)
    graph.add_node("approval", approval)
    graph.add_node("check", check)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "approval", "check"])
    graph.add_edge("tools", "limits")
    graph.add_edge("approval", "agent")
    return graph.compile(checkpointer=InMemorySaver())


def ask(app, question: str, approve: bool = False) -> dict:
    """Run one question to the end; every hand-off gets `approve` as the planner's decision."""
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = app.invoke({"messages": [HumanMessage(question)]}, config)
    handoffs = 0
    while "__interrupt__" in result:
        handoffs += 1
        result = app.invoke(Command(resume={"approved": approve}), config)
    messages = result["messages"]
    return {
        "answer": messages[-1].text,
        "tools": [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls],
        "handoffs": handoffs,
        "approval": result.get("approval"),
        "retries": result.get("retries", 0),
        "messages": messages,
    }
