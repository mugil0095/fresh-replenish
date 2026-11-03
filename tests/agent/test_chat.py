from fakes import Scripted
from langchain_core.messages import AIMessage

from fresh_replenish.agent import chat
from fresh_replenish.agent.graph import build_agent

DAY = {"store_id": 1, "product_id": 10, "date": "2024-06-26"}


def test_chat_shows_tool_calls_and_asks_the_planner(toolbox, capsys):
    call = {"name": "compute_order", "args": {**DAY, "units": 20}, "id": "1"}
    script = [AIMessage("", tool_calls=[call]), AIMessage("20 units needs approval."),
              AIMessage("The planner approved 20 units.")]
    app = build_agent(Scripted(responses=script), toolbox)
    answers = iter(["y"])
    chat.converse(app, "Order 20 units of product 10 at store 1 on 2024-06-26.",
                  ask_planner=lambda prompt: next(answers))
    shown = capsys.readouterr().out
    assert "compute_order" in shown and "approved 20 units" in shown
