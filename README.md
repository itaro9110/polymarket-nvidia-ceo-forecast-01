# Polymarket NVIDIA CEO forecast

Entry for make.fun `polymarket-nvidia-ceo-forecast-01`: forecast every
market of Polymarket's "Next CEO of NVIDIA?" event at every minute of the
4 hours after a checkpoint, as five quantiles (5/25/50/75/95%) of the Yes
price.

## Model

Persistence with an asymmetric, level-dependent spread:

  q_tau(h) = clamp(x + z_tau * S(x) * h^0.45 * mult_tau(x), 0, 1)

- `x` is the market's Yes price at the checkpoint; the median stays at `x`.
- `S(x) = 0.0009 * (1 + 4x)`: the spread widens with the price level.
- `mult = 1 + 6x` for the downside quantiles (tau < 0.5), 1 otherwise:
  high prices crash back down far more often than low prices spike up.
- The spread grows like h^0.45, slower than a random walk's sqrt(h),
  because the jumps mean-revert (flicker episodes bounce and come back).

## Calibration

Constants were tuned on the six practice cases, scored locally against
the next case's prices: local loss 0.330 vs the house baseline's 0.393,
beating it on every practice case. Things tried and rejected: raw
empirical conditional quantiles (flicker-era history overstates the
tails), median drift toward the recent median, and per-market recent
volatility scaling — all scored worse than this model.

## Running

  python3 main.py

Reads `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one
`<case>.out` per `<case>.in` to `$MAKEFUN_OUTPUT_DIR` (default `/output`).
Python 3.13, standard library only, offline, deterministic: two runs on
the same case write identical bytes.
