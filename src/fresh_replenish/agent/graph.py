"""The replenishment agent (spec section 11): one LangGraph agent with four plain-code tools.

agent -> tools -> limit check (a human takes over after 3 failed checks) -> agent
      -> human approval if the last order failed its limits -> number check -> end.
"""

import json
import uuid
from typing import Annotated

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
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
- What-if answers must say they are model-based estimates, not proof of cause and effect.
- If a tool returns an error, say the data is not available; never guess.
- If an order is outside the limits, say so: a planner must approve it.
- Call write_decision_record only when asked to place or record an order.
- Politely refuse anything that is not about ordering fresh food for these stores.
Dates are YYYY-MM-DD. A discount is the price as a share of normal (0.8 means 20% off)."""
REFUSAL = (
    "I can't back every number in my answer with a tool output, so I won't guess. "
    "Please ask a planner."
)
MAX_RETRIES = 2  # answers the number checker may reject before the agent gives up
MAX_FAILED_CHECKS = 3  # failed limit checks before a human takes over


class State(MessagesState):
    failed_checks: int
    retries: int
    approval: str  # "approved" or "rejected" once a planner has decided


class Scripted(FakeMessagesListChatModel):
    """A fake chat model that replays scripted messages, for tests and offline runs."""

    def bind_tools(self, tools, **kwargs):
        return self


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


def make_tools(toolbox: Toolbox) -> list[StructuredTool]:
    def write_decision_record(
        store_id: int, product_id: int, date: str, order: float, reason: str,
        state: Annotated[dict, InjectedState],
    ) -> dict:
        """Save the order, its reason and this conversation's tool outputs."""
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
        orders = _orders(state)
        if orders and not orders[-1]["within_limits"] and not state.get("approval"):
            return "approval"
        return "check"

    def limits(state: State) -> Command:
        failed = sum(not o["within_limits"] for o in _orders(state))
        go = "approval" if failed >= MAX_FAILED_CHECKS and not state.get("approval") else "agent"
        return Command(goto=go, update={"failed_checks": failed})

    def approval(state: State) -> dict:
        decision = interrupt({"order": _orders(state)[-1], "failed_checks": state["failed_checks"]})
        verdict = "approved" if decision.get("approved") else "rejected"
        note = HumanMessage(f"A planner {verdict} this order. {decision.get('note', '')}".strip())
        return {"approval": verdict, "messages": [note]}

    def check(state: State) -> Command:
        sources = [
            m.text for m in state["messages"]
            if isinstance(m, ToolMessage) or (isinstance(m, HumanMessage) and m.name != "checker")
        ]
        bad = unsupported(state["messages"][-1].text, sources)
        if not bad:
            return Command(goto=END)
        retries = state.get("retries", 0)
        if retries >= MAX_RETRIES:
            return Command(goto=END, update={"messages": [AIMessage(REFUSAL)]})
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
    """Run one question to the end; a hand-off gets `approve` as the planner's decision."""
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = app.invoke({"messages": [HumanMessage(question)]}, config)
    handoff = "__interrupt__" in result
    if handoff:
        result = app.invoke(Command(resume={"approved": approve}), config)
    messages = result["messages"]
    return {
        "answer": messages[-1].text,
        "tools": [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls],
        "handoff": handoff,
        "approval": result.get("approval"),
        "retries": result.get("retries", 0),
        "messages": messages,
    }
