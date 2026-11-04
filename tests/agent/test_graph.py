import json

from fakes import Scripted
from langchain_core.messages import AIMessage, HumanMessage

from fresh_replenish.agent.graph import REFUSAL, ask, build_agent

DAY = {"store_id": 1, "product_id": 10, "date": "2024-06-26"}


def call(name, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": {**DAY, **args}, "id": name}])


def run(toolbox, script, question="How much should store 1 order of product 10 on 2024-06-26?"):
    return ask(build_agent(Scripted(responses=script), toolbox), question)


def test_tool_loop_answers_with_tool_numbers(toolbox):
    out = run(toolbox, [call("get_forecast"), AIMessage("Median 5.0, likely 1.0 to 9.0.")])
    assert out["answer"] == "Median 5.0, likely 1.0 to 9.0."
    assert out["tools"] == ["get_forecast"] and not out["handoffs"]


def test_checker_rejects_an_invented_number_then_accepts_a_retry(toolbox):
    out = run(toolbox, [call("compute_order"), AIMessage("Order 12.5."), AIMessage("Order 9.0.")])
    assert out["answer"] == "Order 9.0." and out["retries"] == 1


def test_checker_gives_up_with_a_refusal(toolbox):
    out = run(toolbox, [AIMessage("Order 12.5."), AIMessage("Order 13.5."), AIMessage("14.5")])
    assert out["answer"] == REFUSAL


def test_the_refusal_mentions_an_order_already_saved(toolbox):
    script = [
        call("compute_order"), call("write_decision_record", order=9.0, reason="90% service"),
        AIMessage("Order 12.5."), AIMessage("Order 14.5."), AIMessage("Order 16.5."),
    ]
    out = run(toolbox, script)
    assert out["answer"].startswith(REFUSAL) and "saved" in out["answer"]


def test_over_limit_order_pauses_for_a_human_and_resumes(toolbox):
    script = [
        call("compute_order", units=20), AIMessage("20 units is over the 13.5 limit."),
        AIMessage("A planner rejected the order of 20 units."),
    ]
    out = run(toolbox, script, "Order 20 units of product 10 for store 1 on 2024-06-26.")
    assert out["handoffs"] == 1 and out["approval"] == "rejected"
    assert out["answer"] == "A planner rejected the order of 20 units."


def test_three_failed_limit_checks_hand_off(toolbox):
    script = [call("compute_order", units=u) for u in (20, 18, 15)] + [AIMessage("Handed over.")]
    out = run(toolbox, script)
    assert out["handoffs"] == 1 and out["tools"] == ["compute_order"] * 3


def test_decision_record_gets_the_tool_outputs(toolbox):
    script = [
        call("compute_order"), call("write_decision_record", order=9.0, reason="90% service"),
        AIMessage("Recorded 9.0 units."),
    ]
    run(toolbox, script)
    record = json.loads(toolbox.records.read_text(encoding="utf-8"))
    assert record["tool_outputs"][0]["order"] == 9.0


def test_numbers_in_the_rules_are_not_a_source(toolbox):
    script = [call("get_forecast"), AIMessage("Order 50 units."), AIMessage("Order 9.0 units.")]
    out = run(toolbox, script)
    assert out["retries"] == 1 and out["answer"] == "Order 9.0 units."


def test_percentile_words_from_the_rules_pass_the_checker(toolbox):
    answer = "The 10th to 90th percentile range is 1.0 to 9.0."
    assert run(toolbox, [call("get_forecast"), AIMessage(answer)])["answer"] == answer


def test_an_unapproved_over_limit_order_is_not_recorded(toolbox):
    both = AIMessage("", tool_calls=[
        {"name": "compute_order", "args": {**DAY, "units": 20}, "id": "a"},
        {"name": "write_decision_record", "args": {**DAY, "order": 20, "reason": "x"}, "id": "b"},
    ])
    out = run(toolbox, [both, AIMessage("20 units needs approval."), AIMessage("Rejected: 20.")])
    assert out["handoffs"] == 1 and not toolbox.records.exists()


def test_every_over_limit_order_gets_its_own_hand_off(toolbox):
    script = [
        call("compute_order", units=20), AIMessage("20 needs approval."),
        call("compute_order", units=30), AIMessage("30 needs approval."), AIMessage("Done."),
    ]
    assert run(toolbox, script)["handoffs"] == 2


def test_a_failed_order_is_handed_off_even_when_a_later_one_passes(toolbox):
    script = [
        call("compute_order", units=20), call("compute_order"),
        AIMessage("20 is over the limit; the cost-based order is 9.0."),
        AIMessage("A planner rejected 20; order 9.0."),
    ]
    assert run(toolbox, script)["handoffs"] == 1


def test_the_planner_is_shown_a_failed_order(toolbox):
    calls = [
        {"name": "compute_order", "args": {**DAY, "units": u}, "id": str(u)} for u in (20, 18, 15)
    ]
    calls.append({"name": "compute_order", "args": DAY, "id": "ok"})
    app = build_agent(Scripted(responses=[AIMessage("", tool_calls=calls)]), toolbox)
    result = app.invoke({"messages": [HumanMessage("Order product 10.")]},
                        {"configurable": {"thread_id": "t"}})
    assert result["__interrupt__"][0].value["order"]["within_limits"] is False
