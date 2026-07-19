"""Offline logic tests for the factor backtest. python -m screener.test_factor"""
import numpy as np, pandas as pd
from screener import config as cfg
from screener.backtest_factor import detect_events, forward_and_momentum, summarise

def dates(n): return pd.bdate_range("2022-01-03", periods=n)
passed = 0
def ok(name, cond):
    global passed
    assert cond, f"FAILED: {name}"; passed += 1; print("  ok ", name)

N = 400
rng = np.random.default_rng(0)
r = rng.normal(0, 0.008, N); r[0] = 0
close = 10 * np.cumprod(1 + r)
close[200] = close[199] * 0.82                     # single -18% crash on day 200
for i in range(201, 264):                          # recover +30% over ~3 months
    close[i] = close[200] * (1 + 0.30 * (i - 200) / 63)
for i in range(264, N):
    close[i] = close[263]
idx = dates(N)
close = pd.Series(close, index=idx)
vol = pd.Series(np.full(N, 1e9), index=idx)
bench = pd.Series(np.full(N, 7000.0), index=idx)   # flat market
mcfg = cfg.market_params("ASX")

print("Detection + forward returns:")
evs = detect_events(close, vol, bench, mcfg)
ok("an event is detected", len(evs) >= 1)
ev = [e for e in evs if 198 <= e["pos"] <= 202][0]
ok("raw drop ~ -18%", abs(ev["raw"] - (-0.18)) < 0.02)
ok("window length 1", ev["window_len"] == 1)
fm = forward_and_momentum(close, bench, ev)
ok("3m forward return ~ +30%", fm["fwd_3m"] > 0.25)
ok("3m excess ~ forward (flat market)", abs(fm["excess_3m"] - fm["fwd_3m"]) < 0.01)
ok("6m forward return present & positive", fm["fwd_6m"] is not None and fm["fwd_6m"] > 0.25)

print("Cooldown de-dup:")
c2 = 10 * np.cumprod(1 + rng.normal(0, 0.008, N));
c2[100] = c2[99]*0.82; c2[110] = c2[109]*0.82; c2[300] = c2[299]*0.82   # two close, one far
c2 = pd.Series(c2, index=idx)
evs2 = detect_events(c2, vol, bench, mcfg)
near = [e for e in evs2 if 95 <= e["pos"] <= 130]
ok("two crashes within cooldown -> one event", len(near) == 1)
ok("a separate later crash -> its own event", any(295 <= e["pos"] <= 305 for e in evs2))

print("Summary / bucketing:")
rows = []
for i in range(12):
    rows.append({"market":"ASX","ticker":"T","name":"T","date":ev["date"],
                 "window_len":1,"raw":ev["raw"],"z":ev["z"],
                 "profitable": i % 2 == 0, "roe": 0.03+0.02*i, "de": 0.2+0.1*i,
                 "ey_minus_bond": -0.02+0.01*i, "momentum_6m": -0.2+0.05*i,
                 "fwd_3m": 0.05+0.01*i, "excess_3m": 0.04+0.01*i,
                 "fwd_6m": 0.08+0.01*i, "excess_6m": 0.06+0.01*i})
s = summarise(pd.DataFrame(rows))
ok("summary counts events", s["n_events"] == 12)
ok("baseline 3m stats present", s["baseline_3m"]["n"] == 12 and "hit_rate" in s["baseline_3m"])
ok("profitable split present", "yes" in s["by_metric"]["profitable"] and "no" in s["by_metric"]["profitable"])
ok("roe buckets present", len(s["by_metric"]["roe"]) >= 1)

print(f"\nALL {passed} FACTOR ASSERTIONS PASSED")
