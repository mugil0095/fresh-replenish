# Negative results

What didn't work, or didn't hold, and is reported rather than hidden. The evidence is in [decisions](decisions.md) and the reports.

## Hidden-demand recovery (Phase 3)
- **The deep imputers lost to a simple in-stock-hour profile:**
  - profile: WAPE 37.9%, no bias;
  - SAITS: 45.5%, under-recovers;
  - Tobit: 47.7%, over-recovers by +35.9%;
  - TimesNet: learned to fill hidden hours with zeros (71.0%), no better than doing nothing.
- **Look-ahead in the recovered history:** stockout hours were filled using each series' profile from all 90 days, so the forecast folds' history carries a little information from later weeks. It affects every model alike.

## Forecasting (Phase 4)
- **The foundation models did not beat LightGBM.** Zero-shot, Chronos-2, TimesFM-2.5 and Toto-2.0 trailed by 1.0–1.9 WAPE points, with paired intervals above 0.
- **Every model's median runs 8–13% low.** Demand is right-skewed, so the median sits below the mean, and WAPE rewards the median.

## Calibration and orders (Phase 5)
- **The spec's claim fails across models.** "The most accurate model is not the cheapest" didn't hold: LightGBM's cost-based orders were the cheapest at every cost ratio. It holds only for the decision rule (ordering the median vs ordering by cost), and in 27 of 90 stores, which is too few days per store to test.
- **Calibrated deciles still run 1–5 points low** on the held-out week, whose demand was higher than the weeks the calibration learned from.
- **Availability costs waste:** at the 9× headline ratio, about a third of the units ordered are wasted.

## Service (Phase 6)
- **The Docker image is unverified.** It was checked in CI only, not built locally, and CI runs once the repo is pushed.
- **Category alerts are noisy:** monitoring flags bias, coverage or stockout drift in several categories even in a normal week.

## Agent (Phase 7)
- **Remaining misses:** in 3 of 150 runs the agent didn't record an order when asked, and in 1 run it recorded one nobody asked for.
- **What-ifs are not causal.** Chronos-2's covariate effects can have the "wrong" sign: a 20% discount lowered one item's forecast by 13.8%.
- **Gemini was not evaluated.** The free tier allows 20 requests a day per model, and the run needs about 375. A Google AI Plus subscription doesn't cover API keys.
- **The first evaluation run had a scoring bug:** the scorer ignored the agent's rules text, which made "numbers matching" 52%. Fixed; the reported run is the second.
