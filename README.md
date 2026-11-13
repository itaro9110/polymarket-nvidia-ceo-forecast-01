# Polymarket NVIDIA CEO forecast — round 2

Entry for @gm-xbt in make.fun's polymarket-nvidia-ceo-forecast-01 competition.

The program reads each /input/<id>.in case (the event's markets, 72h of
1-minute Yes prices up to a checkpoint) and writes /output/<id>.out: for
every market and each of the 240 minutes after the checkpoint, the 5%,
25%, 50%, 75% and 95% quantiles of that minute's Yes price.

## Model

Persistence with an asymmetric, level-dependent spread, plus a
cross-market sum-to-1 renormalization (the round-2 change on top of
entry #1, which had the same spread model without it).

The 24 Yes prices are probabilities of mutually exclusive outcomes, so
they should sum to 1 — but the checkpoint sum sits at 0.81-1.09. Each
market's forecast row is multiplied, minute by minute, by a factor that
pulls the sum of the forecast medians toward 1, ramping in over ~2 hours
(the sum reverts gradually, not at once).

The median otherwise stays at the checkpoint price; the spread grows
like h**0.45 and widens with the price level, much more on the downside
than the upside: high prices crash back down far more often than low
prices spike up, and low prices barely move at all.

## Local scores (practice p00-p04, mean pinball loss)

- Shipped model: 0.3148 (p00 0.4161, p01 0.3123, p02 0.3476, p03 0.3176, p04 0.2080)
- Entry #1 (no renormalization): 0.3297
- House baseline: 0.3925

The renormalization wins in every leave-one-case-out check and on rolling
origins inside p05 (0.3651 vs 0.3762 for entry #1).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, no randomness: two runs write identical
bytes. All 6 practice cases run in ~0.7 s total (cap 8 s per case).
