from fresh_replenish.agent.checker import numbers, unsupported

TOOL = '{"date": "2024-06-26", "median": 5.0372, "high": 9.0, "critical_ratio": 0.9}'


def test_numbers_keep_dates_commas_decimals_and_signs():
    text = "Order 1,234.5 units (25.0%) on 2024-06-26, a -3.2 change, from q90 of Chronos-2."
    assert numbers(text) == ["2024-06-26", "1,234.5", "25.0", "-3.2", "2"]


def test_tool_numbers_pass_with_rounding():
    assert unsupported("Median 5.04 (about 5), high 9.0, on 2024-06-26.", [TOOL]) == []


def test_question_numbers_small_counts_and_percent_of_fractions_pass():
    sources = [TOOL, "How much should store 123 order?"]
    assert unsupported("Store 123 orders at the 90% level for 3 days.", sources) == []


def test_unsupported_numbers_and_dates_are_caught():
    assert unsupported("Order 12.5 units on 2024-06-27.", [TOOL]) == ["2024-06-27", "12.5"]
