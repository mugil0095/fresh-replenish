import json

from langchain_core.messages import AIMessage

from fresh_replenish.agent.graph import REFUSAL, Scripted, ask, build_agent

DAY = {"store_id": 1, "product_id": 10, "date": "2024-06-26"}


def call(name, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": {**DAY, **args}, "id": name}])


def run(toolbox, script, question="How much should store 1 order of product 10 on 2024-06-26?"):
    return ask(build_agent(Scripted(responses=script), toolbox), question)


def test_tool_loop_answers_with_tool_numbers(toolbox):
    out = run(toolbox, [call("get_forecast"), AIMessage("Median 5.0, likely 1.0 to 9.0.")])
    assert out["answer"] == "Median 5.0, likely 1.0 to 9.0."
    assert out["tools"] == ["get_forecast"] and not out["handoff"]


def test_checker_rejects_an_invented_number_then_accepts_a_retry(toolbox):
    out = run(toolbox, [call("compute_order"), AIMessage("Order 12.5."), AIMessage("Order 9.0.")])
    assert out["answer"] == "Order 9.0." and out["retries"] == 1


def test_checker_gives_up_with_a_refusal(toolbox):
    out = run(toolbox, [AIMessage("Order 12.5."), AIMessage("Order 13.5."), AIMessage("14.5")])
    assert out["answer"] == REFUSAL


def test_over_limit_order_pauses_for_a_human_and_resumes(toolbox):
    script = [
        call("compute_order", units=20), AIMessage("20 units is over the 13.5 limit."),
        AIMessage("A planner rejected the order of 20 units."),
    ]
    out = run(toolbox, script, "Order 20 units of product 10 for store 1 on 2024-06-26.")
    assert out["handoff"] and out["approval"] == "rejected"
    assert out["answer"] == "A planner rejected the order of 20 units."


def test_three_failed_limit_checks_hand_off(toolbox):
    script = [call("compute_order", units=u) for u in (20, 18, 15)] + [AIMessage("Handed over.")]
    out = run(toolbox, script)
    assert out["handoff"] and out["tools"] == ["compute_order"] * 3


def test_decision_record_gets_the_tool_outputs(toolbox):
    script = [
        call("compute_order"), call("write_decision_record", order=9.0, reason="90% service"),
        AIMessage("Recorded 9.0 units."),
    ]
    run(toolbox, script)
    record = json.loads(toolbox.records.read_text(encoding="utf-8"))
    assert record["tool_outputs"][0]["order"] == 9.0
