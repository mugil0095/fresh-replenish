from fakes import Scripted
from langchain_core.messages import AIMessage

from fresh_replenish.agent import evaluate
from fresh_replenish.agent.graph import build_agent


def test_ollama_cloud_requests_time_out(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    assert evaluate.chat_model("ollama:gpt-oss:120b")._client._client.timeout.read == 120


def test_fifty_questions_of_every_type_from_the_orders(orders):
    questions = evaluate.make_questions(orders)
    assert len(questions) == 50 and questions == evaluate.make_questions(orders)  # seeded
    counts = {t: sum(q["type"] == t for q in questions) for t in evaluate.TYPES}
    assert counts == evaluate.TYPES
    limit = next(q for q in questions if q["type"] == "limit")
    assert limit["handoff"] and limit["tools"] == ["compute_order"]
    assert "write_decision_record" in limit["optional"]  # "place an order" may try to record it


def _factory(toolbox):
    """A scripted agent: a lookup answered from get_forecast, everything else refused."""
    def build(question):
        if question["type"] == "lookup":
            day = {k: question[k] for k in ("store_id", "product_id", "date")}
            call = {"name": "get_forecast", "args": day, "id": "1"}
            answer = "The median is 5.0; the 10th to 90th percentile range is 1.0 to 9.0."
            script = [AIMessage("", tool_calls=[call]), AIMessage(answer)]
        else:
            script = [AIMessage("Sorry, I can only help with fresh-food orders.")]
        return build_agent(Scripted(responses=script), toolbox)

    return build


def test_scores_tool_choice_numbers_and_refusals(toolbox, orders):
    kinds = ("lookup", "out_of_scope")
    questions = [q for q in evaluate.make_questions(orders) if q["type"] in kinds]
    runs, summary = evaluate.evaluate(_factory(toolbox), questions, repeats=2, price=(1.0, 2.0))
    assert len(runs) == 2 * len(questions)
    by_type = summary.set_index("type")
    assert by_type.loc["lookup", "tool_choice"] == 1.0
    assert by_type.loc["out_of_scope", "refusal_correct"] == 1.0
    assert by_type.loc["all", "numbers_match"] == 1.0
    assert "usd_per_run" in summary  # zero here: the scripted model reports no tokens
    assert runs.tools.iloc[0] == "get_forecast" and "90th" in runs.answer.iloc[0]


def test_report_lists_the_headline_measures(toolbox, orders):
    questions = [q for q in evaluate.make_questions(orders) if q["type"] == "lookup"]
    runs, summary = evaluate.evaluate(_factory(toolbox), questions, repeats=1)
    report = evaluate.render_report(summary, "scripted")
    assert report.startswith("# Agent evaluation") and "tool choice" in report


def test_main_without_a_model_names_the_setting(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("FRESH_LLM", raising=False)
    assert evaluate.main(processed_dir=tmp_path, out_dir=tmp_path) == 2
    assert "FRESH_LLM" in capsys.readouterr().err


def test_provider_errors_are_recorded_and_stop_the_run(orders):
    def broken(question):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    questions = evaluate.make_questions(orders)[:5]
    runs, summary = evaluate.evaluate(broken, questions, repeats=3)
    assert len(runs) == evaluate.MAX_ERRORS and runs.error.notna().all()
    assert summary.set_index("type").loc["all", "errors"] == evaluate.MAX_ERRORS


def test_only_limit_questions_use_days_already_over_capacity(orders):
    orders.loc[orders.store_id == 2, "order"] = 300.0  # store 2 plans 600 units a day
    questions = evaluate.make_questions(orders)
    stores = {q["store_id"] for q in questions if q["type"] not in ("limit", "out_of_scope")}
    assert stores == {1}


def test_missing_data_answers_count_with_typographic_apostrophes():
    question = {"id": 1, "type": "missing", "tools": ["get_forecast"], "optional": [],
                "handoff": False}
    out = {"tools": ["get_forecast"], "answer": "Sorry, that forecast isn\u2019t available.",
           "messages": [], "retries": 0, "handoffs": 0}
    assert evaluate.score(question, out, 1.0)["refusal_correct"] == 1.0
