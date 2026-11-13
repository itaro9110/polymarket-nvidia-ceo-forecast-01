"""Forecast Polymarket 'Next CEO of NVIDIA?' Yes prices, 4 hours ahead.

Reads each /input/<id>.in case (the event's markets, 72h of 1-minute Yes
prices up to a checkpoint) and writes /output/<id>.out: for every market
and each of the 240 minutes after the checkpoint, the 5%, 25%, 50%, 75%
and 95% quantiles of that minute's Yes price.

Model: persistence with an asymmetric, level-dependent spread. The median
stays at the checkpoint price; the spread grows like h**0.45 and widens
with the price level, much more on the downside than the upside: high
prices crash back down far more often than low prices spike up, and low
prices barely move at all. Constants were calibrated on the six practice
cases, scored locally against the next case's prices (see BUILD_LOG.md
in the work directory, not part of this entry).
"""
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Spread in price units at h = 1 minute for a market at price 0.
SIGMA0 = 0.0009
# Spread grows like h ** ALPHA (slower than sqrt(h): jumps mean-revert).
ALPHA = 0.45
# Spread scale grows with the price level: 1 + KX * x.
KX = 4.0
# Downside (tau < 0.5) spread grows faster with the level: 1 + KD * x.
KD = 6.0


def _norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("quantile level must be in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02,
         -2.759285984469534e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00)
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
               + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
               + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
           + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r
           + 1.0)
    return num / den


def _price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def forecast_market(x, horizon, zetas):
    """All horizon rows of quantiles for one market at checkpoint price x."""
    scale = SIGMA0 * (1.0 + KX * x)
    down = 1.0 + KD * x
    rows = []
    for h in range(1, horizon + 1):
        base = scale * (h ** ALPHA)
        row = []
        for z in zetas:
            mult = down if z < 0.0 else 1.0
            row.append(_price6(x + z * base * mult))
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(n for n in os.listdir(INPUT_DIR) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r",
                  encoding="utf-8") as f:
            case = json.load(f)
        zetas = [_norm_ppf(t) for t in case["quantiles"]]
        forecasts = []
        for market in case["markets"]:
            prices = market["prices"]
            if not prices:
                raise ValueError("market %s has no prices" % market["id"])
            rows = forecast_market(prices[-1], case["horizon_minutes"],
                                  zetas)
            forecasts.append({"id": market["id"], "q": rows})
        out = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
