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
- **Most of the headline is ordering more.** Against a fixed multiple of the median tuned on past weeks (1.7×), cost-based orders save 6.5% (95% CI 5–8%), not 56%. That is what per-line calibrated ranges are worth at 9×.
- **Availability costs waste:** at the 9× headline ratio, about a third of the units ordered are wasted.

## Second delivery
- **Updating the forecast from the morning's sales barely helped.** In a throwaway run on the past folds at 9× (not kept in the repo), it moved the cost from 0.721 to 0.689. Part of that is circular, so it was dropped. Nearly all the gain comes from counting the shelf.
- **The saving shrinks on observed days:** 32% overall, 23% on days with no stockout. Part of the headline may rest on recovered afternoon demand, and those days are also quieter, so the two effects can't be separated.
- **Waste stays high:** with a one-day shelf life, even a top-up wastes 0.418 units per unit of demand at 9×.
- **It is worth far less if stock keeps:** 24.3 wasted units per store per day at a 1-day shelf life, 6.4 at 2 days and 2.0 at 3 days.
- **First comparisons flattered it, and cost-based orders:** a re-tuned policy was judged against an untuned one. The top-up showed 50–57% savings at 2–3 days instead of 25% and 15%, and cost-based orders showed 78–80% over the median instead of 11% and 8.5% over a tuned multiple.

## Service (Phase 6)
- **Docker is only checked in CI.** The stack runs there on tiny smoke data. It has never been built on this laptop or run on the real outputs.
- **Category alerts are noisy:** monitoring flags bias, coverage or stockout drift in several categories even in a normal week.

## Agent (Phase 7)
- **Record misses, fixed:** after the audit fixes, 5 of 12 "record this order" runs computed the order but never saved it. Stronger wording alone overshot (14 unasked saves), so the save tool is now offered only when the question asks to place, record or save an order; the next run scored 100% on tool choice.
- **Orders placed after a rejection, fixed:** in 6 of 24 over-limit runs, a planner rejected the order and the agent then placed a smaller one on its own, sometimes without mentioning the rejection. Tool calls from the earlier run show the same pattern in 5 of 24. The hand-off score stayed at 100% because the hand-off itself happened; only reading the answers found this. Saving is now blocked once a planner rejects.
- **A fix that made things worse:** the first fix was a rule in the prompt, which every question sees. That run fell to 95% right tools and 92% hand-offs and nearly doubled the tokens, and 2 of 12 record runs never saved (at least one still said "Order recorded"). Moved into the one message that reports a rejection, the scores came back.
- **Unasked saves still happen:** in 2 of 42 "why" and other-ratio runs, the model called the save tool it wasn't offered, up to 18 times in one run. Every call was refused and nothing was saved. These are the 2 tool-choice misses.
- **The stricter checker costs retries, and some answers:** 80% of answers pass the number checker on the first try (96% under the old, looser checker). After retries all 168 pass, but 11 of those (7%) are the agent giving up and telling the user to ask a planner: 7 of 24 "why" runs and 4 of 18 other-ratio runs. The earlier 100% hid 9 such runs out of 150; the evaluation now counts them.
- **Answers after a rejection are short:** all 24 say no order was placed, but only 11 say why.
- **The new tool costs tokens:** 3,392 per run, up from 2,936 (24% more on the same 50 questions), because every call carries one more tool and more rules.
- **What-ifs are not causal.** Chronos-2's covariate effects can have the "wrong" sign: in evaluation question 30, a promotion lowered product 766's median at store 343 from 1.27 to 1.15 units (`reports/tables/agent_runs_gpt-oss-120b.csv`).
- **Gemini was not evaluated.** The free tier allows 20 requests a day per model, and the run needs about 375. A Google AI Plus subscription doesn't cover API keys.
- **The first evaluation run had a scoring bug:** the scorer rejected rule words like "90th percentile", which made "numbers matching" 52%. The fix counted the rules as a source; the reported run is the second.
- **The scores are in-sample.** After the first run, the prompt, the scorer and one question were changed, and the same 50 questions were run again. The same questions were run three more times while adding top-ups. There is no fresh question set.