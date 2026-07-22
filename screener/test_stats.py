"""
Synthetic unit tests for the statistical core. Run: python -m screener.test_stats
Each case constructs a price series with a KNOWN answer and asserts the screen
does the right thing. No network, fully deterministic.
"""
import math
import numpy as np
import pandas as pd

from screener import config as cfg
from screener.stats import evaluate_series, Candidate


def _dates(n):
    return pd.bdate_range("2026-01-01", periods=n)


def _quiet_series(n, daily_vol=0.01, start=10.0, seed=0):
    """A calm stock: small gaussian daily returns, low volatility."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, daily_vol, n)
    rets[0] = 0
    close = start * np.cumprod(1 + rets)
    return pd.Series(close, index=_dates(n))


def _flat_bench(n, start=7000.0):
    # benchmark barely moves (broad market calm)
    rng = np.random.default_rng(99)
    rets = rng.normal(0, 0.004, n)
    rets[0] = 0
    return pd.Series(start * np.cumprod(1 + rets), index=_dates(n))


def _big_vol(n):
    return pd.Series(np.full(n, 1e9), index=_dates(n))  # ample liquidity


N = 140
BENCH = _flat_bench(N)
CAP = 800_000_000  # $800m, clears size gate, benchmarked vs ASX 300


def _apply_crash(close, day_from_end, pct):
    """Multiply price from `day_from_end` (e.g. -1 = last day) onward by (1+pct)."""
    close = close.copy()
    idx = len(close) + day_from_end
    close.iloc[idx:] = close.iloc[idx:] * (1 + pct)
    return close


passed = 0
def check(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok  {name}")


print("Case 1: quiet stock, single -25% crash on last day, flat market -> FLAG")
close = _apply_crash(_quiet_series(N, 0.01, seed=1), -1, -0.25)
c = evaluate_series("AAA", "Alpha", close, _big_vol(N), BENCH, CAP, cfg)
check("candidate returned", isinstance(c, Candidate))
check("raw return ~ -25%", abs(c.raw_return - (-0.25)) < 0.02)
check("z is deeply negative", c.z_score < -3.0)
check("index-relative below -10pp", c.index_relative <= -0.10)
check("event_date == last day", c.event_date == close.index[-1].date().isoformat())
check("benchmark is ASX300 for mid cap", c.benchmark == cfg.BENCHMARK_300)

print("Case 2: high-vol stock, noisy but no window <= -10% -> NO FLAG")
close = _quiet_series(N, 0.05, seed=2)  # 5%/day vol, drifts, no single big drop
c = evaluate_series("BBB", "Beta", close, _big_vol(N), BENCH, CAP, cfg)
check("no candidate (no >=10% window)", c is None)

print("Case 3: -25% drop but market ALSO -25% (broad selloff) -> NO FLAG (cond2)")
close = _apply_crash(_quiet_series(N, 0.01, seed=3), -1, -0.25)
bench_crash = _apply_crash(BENCH.copy(), -1, -0.25)
c = evaluate_series("CCC", "Gamma", close, _big_vol(N), bench_crash, CAP, cfg)
check("no candidate (index-relative not extreme)", c is None)

print("Case 4: quiet -25% crash but THIN volume -> NO FLAG (liquidity)")
close = _apply_crash(_quiet_series(N, 0.01, seed=4), -1, -0.25)
thin_vol = pd.Series(np.full(N, 100.0), index=_dates(N))  # 100 shares * ~$10 = ~$1k/day
c = evaluate_series("DDD", "Delta", close, thin_vol, BENCH, CAP, cfg)
check("no candidate (below liquidity floor)", c is None)

print("Case 5: quiet -25% crash but SUB-$100m cap -> NO FLAG (size)")
close = _apply_crash(_quiet_series(N, 0.01, seed=5), -1, -0.25)
c = evaluate_series("EEE", "Eps", close, _big_vol(N), BENCH, 50_000_000, cfg)
check("no candidate (below size floor)", c is None)

print("Case 6: 3-day cumulative -15% slide (no single big day) -> FLAG on window")
base = _quiet_series(N, 0.012, seed=6)
base = _apply_crash(base, -3, -0.09)
base = _apply_crash(base, -2, -0.09)
base = _apply_crash(base, -1, -0.09)  # compounding ~ -25% over 3 days (clears 20% floor)
c = evaluate_series("FFF", "Zeta", base, _big_vol(N), BENCH, CAP, cfg)
check("candidate returned for multi-day slide", isinstance(c, Candidate))
check("window length between 3 and 5", 3 <= c.window_len <= 5)
check("raw drop >= 10%", c.raw_return <= -0.10)

print("Case 7: dedup identity stable as window rolls forward one day")
# Same crash observed on day t, then re-observed on day t+1 (one more calm day).
close_t = _apply_crash(_quiet_series(N, 0.01, seed=7), -2, -0.25)   # crash 2nd-last day
close_t = close_t.iloc[:-1]                                          # 'today' = crash day
c_t = evaluate_series("GGG", "Eta", close_t, _big_vol(N-1), BENCH.iloc[:-1], CAP, cfg)
close_t1 = _apply_crash(_quiet_series(N, 0.01, seed=7), -2, -0.25)  # +1 calm day after
c_t1 = evaluate_series("GGG", "Eta", close_t1, _big_vol(N), BENCH, CAP, cfg)
check("both days flag", c_t is not None and c_t1 is not None)
check("SAME dedup key across the two scan days", c_t.key() == c_t1.key())

print("Case 8: US market params flow through the (unchanged) stats core")
us = cfg.market_params("US")
check("US large benchmark = S&P 500", us.BENCHMARK_200 == "^GSPC")
check("US small benchmark = Russell 2000", us.BENCHMARK_300 == "^RUT")
check("US size floor US$1bn", us.MIN_MARKET_CAP == 1_000_000_000)
closeu = _apply_crash(_quiet_series(N, 0.012, seed=8), -1, -0.25)
cu = evaluate_series("IBM", "Intl Business Machines", closeu, _big_vol(N), BENCH, 30_000_000_000, us)
check("US candidate returned", isinstance(cu, Candidate))
check("tagged market US", cu.market == "US")
check("currency USD", cu.currency == "USD")
check("US mega-cap benchmarked vs S&P 500", cu.benchmark == "^GSPC")
check("dedup key namespaced by market", cu.key().startswith("US:IBM:"))

print(f"\nALL {passed} ASSERTIONS PASSED")
