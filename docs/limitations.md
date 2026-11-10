# Limitations

What this project can and can't claim. Results that failed or didn't hold are in [negative results](negative_results.md).

- **Simulated, not live.** Every cost figure comes from one held-out week simulated offline, with no A/B test or live trial.
- **Recovered demand is an estimate.** The simulation scores orders against recovered demand, not true demand, which nobody observes on stockout days. Recovery was validated on in-stock hours hidden on purpose.
- **Costs are assumed.** The data has no prices, so a lost sale is assumed to cost 0.5× to 9× a wasted unit. The −56% headline is at 9×; at equal costs, ordering by cost and ordering the forecast cost about the same. It is also measured against ordering the median itself: a fixed 1.7× the median, tuned on past weeks, gets within 6.5%.
- **Relative units.** Sales are normalised, so results are percentages and ratios, not money or units.
- **Simple ordering.** No lead time for the morning order, pack sizes, minimum order quantities or supplier constraints.
- **Shelf life is assumed.** The data has none. The main results throw unsold stock away the same night. The check with 2 and 3 days gives every product the same life, only at 9×, and assumes older stock sells as well as fresh at full price. The API serves the one-day plan.
- **The second delivery is simulated too.** Its −32% leaves out the delivery's own cost, assumes a 2-hour lead time and an exact shelf count, and partly rests on recovered afternoon demand (−23% on days with no stockout). Which stores earn it back is measured on 90 stores and one held-out week, with the delivery's cost tried as a range because the data has none. It also assumes a one-day shelf life: the delivery is worth 24.3 wasted units per store per day at 1 day, 6.4 at 2 and 2.0 at 3. `/topup` trusts the count and the clock: its targets assume one 13:00 count a day. A later count over-orders, and a second count before the delivery lands orders the same units again, because the API keeps no record of what was ordered.
- **Limits are made up.** The agent's line limit (1.5 × the 90th percentile) and the 400-unit store-day capacity are assumptions, not store rules.
- **What-ifs aren't causal.** Discounts and promotions weren't assigned at random, so Chronos-2 scenario answers are model estimates.
- **Most agent scores are in-sample.** The prompt and the save rule were tuned on the same 56 questions they're scored on, over repeated runs, with one model (`gpt-oss:120b`). A held-out set of 56 reworded questions, run once, scored about the same (99% right tools, 23 of 24 hand-offs), but it shares the model, the week and the scorer. The save tool is offered only when the question says place, record or save.
- **The number checker reads percentile labels by pattern.** The patterns come from how `gpt-oss:120b` types them, so another model may trigger retries or give-ups.
- **The agent stops after 5 failed tool calls.** A question about 5 or more missing products ends in "ask a planner", not "no data". Loops of successful calls aren't capped.
- **The agent's top-ups are advice.** It saves nothing, skips the order limits and store capacity, and assumes the shelf was counted at the decision hour (13:00), as `/topup` does. It doesn't mention the morning share.
- **One dataset, one retailer.** All results come from FreshRetailNet-50K (Dingdong, 18 Chinese cities, spring 2024).
- **Not production-hardened.** No authentication on the API, one process, and the orders file is the only store. CI runs Docker Compose on tiny smoke data, but the stack has not been run on the real outputs.
