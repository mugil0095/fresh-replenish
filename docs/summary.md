# Fresh-food ordering for dark stores: one-page summary

**Ordering by cost, not by the most accurate forecast, cut the simulated cost of waste and lost sales by 56%.**
- The test was a held-out week: 4,928 store-product lines (538 products) in 90 quick-commerce dark stores.
- It assumes a lost sale costs 9× a wasted unit, as availability-first quick commerce runs today.
- The data has no prices, so the cost is assumed, and other ratios are shown too.
- Most of the gain comes from ordering more than the forecast: a fixed 1.7× the forecast, tuned on past weeks, gets within 6.5%.

## The problem

- **Fresh food fails two ways:** what doesn't sell spoils, and what sells out loses the sale and often the customer.
- **Stockouts are common:** a product runs out at some point on 44% of its days, and by 9 pm 41% of items are out of stock.
- **Sales records understate demand:** a sold-out item records only what was on the shelf. Forecasts trained on that data order too little, which causes more stockouts, which hides more demand.

## What was built

1. **Recover hidden demand:** fill in what customers would have bought during stockout hours.
2. **Forecast** each product's demand a week ahead as a range, not a single number. LightGBM was compared with Amazon's, Google's and Datadog's AI forecasting models.
3. **Order by cost:** each store orders the amount that minimises waste plus lost sales, given what each one costs.
4. **Serve and explain:**
   - a nightly job writes orders for all 50,000 store-product lines in 13.3 minutes;
   - an API serves them in about 15 ms, and turns a store's midday shelf count into a top-up order;
   - an AI agent explains orders, advises top-ups, answers "what if we discount this?", and sends unusual orders to a human.

## What we found

| Finding | Number |
|---|---|
| Real demand hidden by stockouts | +28% overall, +62% on days with a stockout |
| Ordering by cost vs ordering the most accurate forecast | −56% cost; served 96% of demand vs 77% |
| Ordering by cost vs reordering last week's demand | −55% cost |
| Adding a midday top-up delivery | a further −31% cost across all 50,000 lines (−23% on days with fully observed demand), before the delivery's own cost |
| Ordering by cost vs a tuned 1.7× the forecast | −6.5% cost (95% CI 5–8%) |
| If unsold stock keeps 2–3 days instead of 1 | cost falls by two-thirds or more; ordering by cost keeps an 8–11% edge |
| Forecast ranges that hold on a new week | 80% of demand fell inside the stated 80% range |
| AI agent, 56 scripted questions run 3 times | right tools 99%; every answer passed the number check, though 7% gave up and told the user to ask a planner; every risky order sent to a human |

- **The most accurate forecast is not the best order.** At 9×, ordering the "most likely" amount runs out on 56% of days and costs more than simply reordering last week's demand. Ordering by cost is the cheapest choice at every cost ratio tested.
- **A trained LightGBM beat the AI foundation models,** but only narrowly (1–2 points of error). The foundation models need no training, which makes them a fast start for new stores.

## What it means

- **Set the service level from costs.** Each product's order should come from its forecast range at the level its waste and lost-sale costs justify, not from the single best guess.
- **Recover hidden demand before forecasting.** Otherwise every stockout makes the next order smaller.
- **Start with a simple multiple.** A fixed multiple of the forecast captures most of the gain. Per-line ranges add another 5–8%.
- **Price a second delivery.** At the assumed 9×, counting the shelf at 13:00 and topping up by 15:00 pays if a delivery costs less than about 24 wasted units per store per day. That is for stock that lasts one day: at 2 days the figure falls to about 6, and at 3 days to about 2.
- **Keep a human on unusual orders.** The agent explains the routine ones and routes the exceptions.

## Caveats

- **Costs:** they are assumed, and results change with the cost ratio. At equal costs, ordering by cost and ordering the forecast cost about the same.
- **Simulation:** this is one held-out week, simulated against recovered (estimated) demand, not a live trial.
- **Shelf life:** the data has none. The main results assume unsold stock is thrown away the same night, and a check with 2 and 3 days gives every product the same life.
- **Scale:** sales in the data are normalised, so results are relative (%), not money.
