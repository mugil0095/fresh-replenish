# Fresh-food ordering for dark stores: one-page summary

**Ordering by cost, not by the most accurate forecast, cut the simulated cost of waste and lost sales by 56%.**
- The test was a held-out week: 4,928 products in 90 quick-commerce dark stores.
- It assumes a lost sale costs 9× a wasted unit, as availability-first quick commerce runs today.
- The data has no prices, so the cost is assumed, and other ratios are shown too.

## The problem

- **Fresh food fails two ways:** what doesn't sell spoils, and what sells out loses the sale and often the customer.
- **Stockouts are common:** 44% of store-days run out of something, and by 9 pm 41% of items are out of stock.
- **Sales records understate demand:** a sold-out item records only what was on the shelf. Forecasts trained on that data order too little, which causes more stockouts, which hides more demand.

## What was built

1. **Recover hidden demand:** fill in what customers would have bought during stockout hours.
2. **Forecast** each product's demand a week ahead as a range, not a single number. LightGBM was compared with Amazon's, Google's and Datadog's AI forecasting models.
3. **Order by cost:** each store orders the amount that minimises waste plus lost sales, given what each one costs.
4. **Serve and explain:**
   - a nightly job writes orders for all 50,000 products in 11.5 minutes;
   - an API serves them in 7 ms;
   - an AI agent explains orders, answers "what if we discount this?", and sends unusual orders to a human.

## What we found

| Finding | Number |
|---|---|
| Real demand hidden by stockouts | +28% overall, +62% on days with a stockout |
| Ordering by cost vs ordering the most accurate forecast | −56% cost; served 96% of demand vs 77% |
| Ordering by cost vs reordering last week's sales | −55% cost |
| Forecast ranges that hold on a new week | 80% of demand fell inside the stated 80% range |
| AI agent, 150 scripted questions | right tools 97%; every number traceable; every risky order sent to a human |

- **The most accurate forecast is not the best order.** When a lost sale costs more than a wasted unit, ordering the "most likely" amount runs out on 57% of days. That costs more than simply reordering last week's sales.
- **A trained LightGBM beat the AI foundation models,** but only narrowly (1–2 points of error). The foundation models need no training, which makes them a fast start for new stores.

## What it means

- **Set the service level from costs.** Each product's order should come from its forecast range at the level its waste and lost-sale costs justify, not from the single best guess.
- **Recover hidden demand before forecasting.** Otherwise every stockout makes the next order smaller.
- **Keep a human on unusual orders.** The agent explains the routine ones and routes the exceptions.

## Caveats

- **Costs:** they are assumed, and results change with the cost ratio. At equal costs, ordering by cost and ordering the forecast cost about the same.
- **Simulation:** this is one held-out week, simulated with recorded demand, not a live trial.
- **Scale:** sales in the data are normalised, so results are relative (%), not money.
