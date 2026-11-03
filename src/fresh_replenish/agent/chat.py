"""Ask the agent one question from the command line; you play the planner on any hand-off.

For example:
    uv run python -m fresh_replenish.agent.chat "Place an order of 6 units of product 4 for store
    57 on 2024-07-02."
The model comes from FRESH_LLM (default ollama:gpt-oss:120b); keys come from `.env`.
"""

import os
import sys
import uuid

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from fresh_replenish.agent.evaluate import chat_model
from fresh_replenish.agent.graph import build_agent
from fresh_replenish.agent.tools import Toolbox, chronos_scenario
from fresh_replenish.config import PROCESSED_DIR, RAW_DIR, REPO_ROOT
from fresh_replenish.data.db import connect
from fresh_replenish.data.download import local_paths


def converse(app, question: str, ask_planner=input) -> None:
    """Print each tool call and the answer; a hand-off asks the planner to approve."""
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    result = app.invoke({"messages": [HumanMessage(question)]}, config)
    while "__interrupt__" in result:
        order = result["__interrupt__"][0].value["order"]
        print(f"\nHand-off: {order['order']} units is outside the limits "
              f"(line limit {order['line_limit']}, store-day total {order['store_day_total']}).")
        approved = ask_planner("Approve this order? [y/N] ").strip().lower() == "y"
        result = app.invoke(Command(resume={"approved": approved}), config)
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"tool: {call['name']}({call['args']})")
    print("\n" + result["messages"][-1].text)


def main(question: str) -> int:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    paths = local_paths(RAW_DIR)
    if paths is None:
        return 2
    con = connect(*paths)
    toolbox = Toolbox(
        pd.read_parquet(PROCESSED_DIR / "orders.parquet"),
        scenario=chronos_scenario(con, PROCESSED_DIR / "recovered_demand.parquet"),
    )
    model = chat_model(os.environ.get("FRESH_LLM", "ollama:gpt-oss:120b"))
    converse(build_agent(model, toolbox), question)
    return 0


if __name__ == "__main__":
    default = "How much should store 78 order of product 638 on 2024-06-29, and why?"
    sys.exit(main(" ".join(sys.argv[1:]) or default))
