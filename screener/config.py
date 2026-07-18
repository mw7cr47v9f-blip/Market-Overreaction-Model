"""
Configuration for the multi-market overreaction screen.

Global statistical thresholds are shared across markets; everything that differs
by market (universe source, benchmarks, size/liquidity floors, currency, and the
primary-source announcement feed) lives in MARKETS. `market_params(name)` merges
the two into one namespace with the attribute names stats.py expects, so the
statistical core stays completely market-agnostic.
"""
from types import SimpleNamespace

# ---- Global statistical thresholds (same in every market) ----------------

Z_THRESHOLD = -2.5           # window return <= this many SD below zero
INDEX_REL_THRESHOLD = -0.10  # underperform benchmark by >= 10pp
ABS_DROP_THRESHOLD = -0.10   # raw decline >= 10%
WINDOW_LENGTHS = [1, 2, 3, 4, 5]
VOL_LOOKBACK = 90            # trailing trading days for the volatility baseline
MIN_VOL_OBS = 40            # min trailing returns to trust the vol estimate
HISTORY_CALENDAR_DAYS = 200

# ---- Per-market settings -------------------------------------------------

MARKETS = {
    "ASX": {
        "suffix": ".AX",
        "currency": "AUD",
        "min_market_cap": 100_000_000,       # >= A$100m
        "min_avg_daily_value": 150_000,      # A$/day liquidity floor
        "large_cap_cutoff": 5_000_000_000,   # >= A$5bn benchmarks vs ASX 200
        "benchmark_large": "^AXJO",          # S&P/ASX 200
        "benchmark_small": "^AXKO",          # S&P/ASX 300
        "universe": "asx_directory",
        "announcements": "asx",
    },
    "US": {
        "suffix": "",
        "currency": "USD",
        # Universe is the S&P 1500 (large+mid+small); its members already sit
        # comfortably above US$1bn, so we set the floor there and skip the long
        # tail of micro-caps (keeps the daily yfinance pull tractable). Flagged.
        "min_market_cap": 1_000_000_000,     # >= US$1bn
        "min_avg_daily_value": 2_000_000,    # US$/day liquidity floor (US trades heavier)
        "large_cap_cutoff": 20_000_000_000,  # >= US$20bn benchmarks vs S&P 500
        "benchmark_large": "^GSPC",          # S&P 500
        "benchmark_small": "^RUT",           # Russell 2000
        "universe": "sp1500",
        "announcements": "sec",
    },
}

# ASX module-level defaults kept so the config module itself doubles as an ASX
# params object (used by the unit tests).
_m = MARKETS["ASX"]
MARKET = "ASX"
SUFFIX = _m["suffix"]
CURRENCY = _m["currency"]
MIN_MARKET_CAP = _m["min_market_cap"]
MIN_AVG_DAILY_VALUE = _m["min_avg_daily_value"]
LARGE_CAP_CUTOFF = _m["large_cap_cutoff"]
BENCHMARK_200 = _m["benchmark_large"]
BENCHMARK_300 = _m["benchmark_small"]

_GLOBALS = ["Z_THRESHOLD", "INDEX_REL_THRESHOLD", "ABS_DROP_THRESHOLD",
            "WINDOW_LENGTHS", "VOL_LOOKBACK", "MIN_VOL_OBS", "HISTORY_CALENDAR_DAYS"]


def market_params(name: str) -> SimpleNamespace:
    """Return a namespace carrying every attribute stats.py reads, for `name`.
    BENCHMARK_200/300 are reused as the large/small benchmark slots regardless
    of market so the statistical core needs no changes."""
    m = MARKETS[name]
    ns = SimpleNamespace()
    for k in _GLOBALS:
        setattr(ns, k, globals()[k])
    ns.MARKET = name
    ns.SUFFIX = m["suffix"]
    ns.CURRENCY = m["currency"]
    ns.MIN_MARKET_CAP = m["min_market_cap"]
    ns.MIN_AVG_DAILY_VALUE = m["min_avg_daily_value"]
    ns.LARGE_CAP_CUTOFF = m["large_cap_cutoff"]
    ns.BENCHMARK_200 = m["benchmark_large"]
    ns.BENCHMARK_300 = m["benchmark_small"]
    ns.UNIVERSE = m["universe"]
    ns.ANNOUNCEMENTS = m["announcements"]
    return ns
