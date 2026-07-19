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

print("Entry timing (dead-cat filter):")
from screener.backtest_factor import find_entry
# after signal at i=5: day6 is a weak up-tick (dead cat), day9 is a real break on volume
ec = pd.Series([10, 9, 8, 8.2, 7.5, 7.0, 7.1, 6.8, 6.9, 8.0, 8.5])
ev_ = pd.Series([1000]*11); ev_.iloc[9] = 5000
ok("naive rule buys the dead-cat pop (day 6)", find_entry(ec, ev_, 5, "up1") == 6)
ok("confirmed rule waits for the real break (day 9)", find_entry(ec, ev_, 5, "confirmed") == 9)
falling = pd.Series([10, 9.5, 9, 8.6, 8.3, 8.0, 7.7, 7.5])  # never breaks up
ok("confirmed rule skips a persistent faller (None)", find_entry(falling, pd.Series([1000]*8), 2, "confirmed", max_wait=5) is None)

print("Exit rules (recover-to-pre-drop):")
from screener.backtest_factor import exit_rules
xc = pd.Series([10.0]*8 + [10.0, 8.5, 7.0, 7.5, 8.0, 8.8, 9.5, 10.2] + [10.2]*14)  # pre=10, entry=7, recovers at pos15
xr = exit_rules(xc, {"pos": 10, "window_len": 2})
ok("recovered to pre-drop", xr["recovered"] is True)
ok("days_to_recover == 5", xr["days_to_recover"] == 5)
ok("target return ~ pre/entry-1 (+42.9%)", abs(xr["r_target"] - (10/7 - 1)) < 0.01)
never = pd.Series([10.0]*8 + [10.0, 9.0, 7.0] + [7.0]*20)  # entry 7, never gets back to 10
xr2 = exit_rules(never, {"pos": 10, "window_len": 2})
ok("never-recover flagged", xr2["recovered"] is False)
sl = pd.Series([10.0]*8 + [10.0, 9.0, 8.0, 7.2] + [7.2]*30)  # entry 8, falls to 7.2 (-10%)
xr3 = exit_rules(sl, {"pos": 10, "window_len": 2})
ok("8% stop-loss fires on the further drop", abs(xr3["r_sl8"] - (7.2/8 - 1)) < 0.01)

print("Quality factors (npat margin, fcf, growth, cagr):")
from screener.backtest_factor import _cagr, metrics_for_event
ok("cagr 10%", _cagr([100, 110, 121]) == 0.1)
ok("cagr needs >=2 points", _cagr([100]) is None)
ok("cagr uses real year-gap past a loss year", _cagr([100, -5, 120]) == round((120/100)**0.5 - 1, 4))
_fund = {"fy": [{"date": pd.Timestamp("2023-06-30"), "ni": 100, "eq": 500, "debt": 100, "rev": 1000, "eps": 1.0, "fcf": 80},
                {"date": pd.Timestamp("2024-06-30"), "ni": 150, "eq": 600, "debt": 100, "rev": 1200, "eps": 1.5, "fcf": 120}],
         "shares_out": 100, "sector": "Information Technology", "dividends": None}
_m = metrics_for_event(_fund, "2025-01-01", 50.0, "US")
ok("npat_margin = latest ni/rev", _m["npat_margin"] == round(150/1200, 4))
ok("fcf_margin = latest fcf/rev", _m["fcf_margin"] == round(120/1200, 4))
ok("rev_growth CAGR", _m["rev_growth"] == round(1200/1000 - 1, 4))
ok("eps_growth CAGR", _m["eps_growth"] == round(1.5/1.0 - 1, 4))

print(f"\nALL {passed} FACTOR ASSERTIONS PASSED")
