# Polymarket 'Next CEO of NVIDIA?' forecast — round 3

Forecasts each market's Yes price at every minute of the 4 hours after
a checkpoint: the 5%, 25%, 50%, 75% and 95% quantiles. Pure Python 3.13
stdlib, offline, deterministic.

## Model

Persistence with an asymmetric, level-dependent spread, plus three
round-3 refinements on top of the round-2 entry:

- **Sum-to-1 renormalization.** The 24 Yes prices are probabilities of
  mutually exclusive outcomes, so they should sum to 1; the checkpoint
  sum sits at 0.81-1.09. Each forecast row is multiplied by a factor
  pulling the median sum toward 1, ramping in over ~2 hours.
- **Jump-aware spreads.** Prices sit flat most minutes then flicker-
  jump. A market that jumped (|1-min change| > 0.05) in the last hour
  gets its spread widened by up to 2x, decaying over ~2 hours; a
  further widening comes from its jump count over the last 4 hours.
- **Tighter inner quantiles.** The 25%/75% quantiles are 15% narrower
  and grow more slowly with horizon (h^0.40 vs h^0.45) than the outer
  5%/95% tails, which are 10% wider.

The median stays at the checkpoint price; the spread grows like
h^0.45, wider for higher prices and on the downside (high prices crash
back down far more often than low prices spike up).

## Local practice (p00-p04, scored against the next case's prices)

| model | mean pinball loss |
|---|---|
| house baseline | 0.392518 |
| entry #1 (persistence + spread) | 0.329740 |
| entry #2 (+ sum-to-1 renorm) | 0.314827 |
| **this entry (round 3)** | **0.304798** |

Beats entry #2 on every practice case (p00 0.3942, p01 0.3057,
p02 0.3463, p03 0.3065, p04 0.1988); leave-one-case-out wins 5/5;
rolling origins inside p05 win on all 6 windows (0.3502 vs 0.3651);
the frozen-price stress test is also better than entry #2 (0.0813 vs
0.0980). What did NOT work: cross-market spillover drift (the renorm
already captures the level coupling — extra drift double-counts),
spike-based downside widening, and per-market activity scaling.
