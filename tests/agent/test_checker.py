import pytest

from fresh_replenish.agent.checker import numbers, unsupported

TOOL = '{"date": "2024-06-26", "median": 5.0372, "high": 9.0, "critical_ratio": 0.9}'


def test_numbers_keep_dates_commas_decimals_and_signs():
    text = "Order 1,234.5 units (25.0%) on 2024-06-26, a -3.2 change, from q90 of Chronos-2."
    assert numbers(text) == ["2024-06-26", "1,234.5", "25.0%", "-3.2", "2"]


def test_tool_numbers_pass_with_rounding():
    assert unsupported("Median 5.04 (about 5), high 9.0, on 2024-06-26.", [TOOL]) == []


def test_question_numbers_small_counts_and_percent_of_fractions_pass():
    sources = [TOOL, "How much should store 123 order?"]
    assert unsupported("Store 123 orders at the 90% level for 3 days.", sources) == []


def test_unsupported_numbers_and_dates_are_caught():
    assert unsupported("Order 12.5 units on 2024-06-27.", [TOOL]) == ["2024-06-27", "12.5"]


def test_typographic_dashes_and_spaces_are_read_like_plain_ones():
    answer = "On 2024‑06‑26 the range is 1.0–9.0 (median 5.04)."
    assert numbers(answer) == ["2024-06-26", "1.0", "9.0", "5.04"]
    assert unsupported(answer, [TOOL]) == ["1.0"]  # 1.0 is not in the tool output


def test_a_negative_result_may_be_stated_as_a_fall():
    assert unsupported("Demand falls by 13.8%.", ['{"change_pct": -13.8}']) == []


def test_a_fraction_counts_as_a_percentage_only_with_a_percent_sign():
    assert unsupported("At the 90% level, order 90 units.", [TOOL]) == ["90"]
    assert unsupported("A 90 percent (or 90-percent) level.", [TOOL]) == []


def test_ranks_are_not_quantities():
    assert unsupported("Between the 25th and 75th percentile.", [TOOL]) == []


# As gpt-oss:120b wrote them: at a ratio of 1 no tool output holds 0.9, so these are labels only.
@pytest.mark.parametrize("label", [
    "(10-th - 90-th percentile)", "(10 th - 90 th percentile)", "the 90 th-percentile value",
    "(10 - 90 percentile)", "the 90-percentile value", "(10 % - 90 % percentile)",
    "(10 %-90 % percentile)", "(90 % percentile)", "(10 %-90 %)", "(10-th % - 90-th %)",
    "Low (10 %), median (50 %), high (90 %)", "Median (50 th)", "**High (90\u202f%): 1.03**",
])
def test_percentile_labels_are_ranks_however_they_are_typed(label):
    assert unsupported(f"{label}: 5.04.", ['{"median": 5.0372, "high": 1.03, "ratio": 1.0}']) == []


@pytest.mark.parametrize("answer", [
    "Order 90 units.", "Order 10-90 units.", "Demand rises 50 %.", "Demand rises 50%.",
    "Sales fall 90 % next week.", "The 90 percent level.", "Demand is high (75 %).",
    "Sales rise from 10 to 90 %.",
])
def test_a_quantity_that_looks_like_a_percentile_is_still_rejected(answer):
    assert unsupported(answer, ['{"median": 5.0372, "ratio": 1.0}'])


def test_a_percent_with_a_space_counts_like_one_without():
    assert unsupported("Demand falls 13.8 %.", ['{"change_pct": -13.8}']) == []
    assert unsupported("Orders at the 90 % level.", [TOOL]) == []


def test_numbers_written_as_words_are_caught():
    assert unsupported("Order twelve units.", [TOOL]) == ["twelve"]
    assert unsupported("Order twelve units.", [TOOL, "Can store 1 order twelve?"]) == []
