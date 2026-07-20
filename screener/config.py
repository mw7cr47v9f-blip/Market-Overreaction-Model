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
ABS_DROP_THRESHOLD = -0.15   # raw decline >= 15% (tightened from 10%: backtest showed
                             # the edge is concentrated in deeper drops — 10-15% falls
                             # barely beat the market, 15-30% falls win ~65-70%)
MAX_DROP_FLOOR = -0.60       # ignore drops WORSE than this. A >60% fall in <=5 days is
                             # a solvency event / delisting / data discontinuity, not an
                             # overreaction to bad news. The survivorship-free backtest
                             # showed these near-100% "drops" both destroy the return
                             # maths (near-zero entry -> Infinity fwd return) and are
                             # untradeable. Applies to live screen AND backtest.
WINDOW_LENGTHS = [1, 2, 3, 4, 5]
VOL_LOOKBACK = 90            # trailing trading days for the volatility baseline
MIN_VOL_OBS = 40            # min trailing returns to trust the vol estimate
HISTORY_CALENDAR_DAYS = 200

# ---- Sector tilt (the locked model favours these, hard-excludes the avoids) --
# Backtest verdict: favoured-only book had the best risk-adjusted return (Sharpe
# ~1.5); the avoid-sectors LOST to the market at equal volatility. See methodology.
import re as _re
# "technology" (bare) covers S&P "Information Technology", ASX "Technology Hardware
# & Equipment" AND Nasdaq's plain "Technology" label — the three feeds name it
# differently, so match the common root.
FAVOURED_SECTORS_RE = (r"\btechnology|software|semiconductor|"
                       r"discretionary|consumer cyclical|industrial|capital goods|automobile")
# Note: EODHD tags consumer-discretionary names as "Consumer Cyclical" (not the S&P
# "Consumer Discretionary"), so both labels are matched. "consumer cyclical" is written
# out in full so it does NOT match "Consumer Non-Cyclicals" (staples), which stay unfavoured.
AVOID_SECTORS_RE = r"material|pharma|biotech|telecom|real estate"
HOLD_MONTHS = 3             # locked time-based exit; no price stop-loss

# ---- Profitability gate (added from the survivorship-free factor study) ------
# The clean 2012-2026 NYSE backtest showed the bottom third by NPAT margin AND by
# FCF margin (the deeply loss-making, cash-burning names) had ~50% hit / ~0 excess
# — dead weight. Excluding them lifts the model's hit rate ~58.6% -> ~64.4% and,
# crucially, makes the edge hold up in calm markets (2022-26 excess +0.3% -> +2.8%).
# Thresholds are the ~33rd-percentile cutoffs, rounded: reject clearly unprofitable
# names, keep break-even-or-better. Set REQUIRE_QUALITY=False to disable the gate.
REQUIRE_QUALITY = True
NPAT_MARGIN_MIN = -0.05     # net profit margin floor (bottom-tercile cut ~ -6%)
FCF_MARGIN_MIN = 0.0        # free-cash-flow margin floor (bottom-tercile cut ~ -1%)


def is_favoured(sector) -> bool:
    return bool(_re.search(FAVOURED_SECTORS_RE, str(sector).lower()))


def is_avoided(sector) -> bool:
    return bool(_re.search(AVOID_SECTORS_RE, str(sector).lower()))


# ---- Trigger-type filter (from the 8-K trigger study) ------------------------
# Drops caused by structural repricing — a lost contract, an impairment/write-down,
# or distress (bankruptcy/delisting/restatement) — do NOT mean-revert: hit rate 38%,
# excess -10%/trade, 38% lose >20% (vs 16% for sentiment drops). Rare (~0.2% of
# events) but toxic, so we hard-exclude them. Sentiment/other catalysts are kept.
# Fails OPEN: only a POSITIVELY-identified structural 8-K excludes a name; an event
# with no 8-K ("none") is kept, since those recover fine and are the majority.
EXCLUDE_STRUCTURAL_TRIGGERS = True
STRUCTURAL_TRIGGERS = ("contract_loss", "cost_impair", "distress")


def is_bad_trigger(trigger_primary) -> bool:
    return bool(EXCLUDE_STRUCTURAL_TRIGGERS and trigger_primary in STRUCTURAL_TRIGGERS)


# ---- Director-buying filter + new exits (staged; validated in the re-run first) ----
# Prior-6-month director open-market buying lifted the gated cohort to ~70% hit /
# Sharpe ~1.5. Adopted as a HARD filter (Keegan's call) — accepting a smaller book
# now, to be rebroadened by adding TSX/LSE (and revisiting ASX) under the new rules.
# Kept False until the combined backtest (confirmed entry + trailing stop + time exit
# + this filter) is confirmed and the live insider fetch is wired.
REQUIRE_DIRECTOR_BUY = False
USE_TRAILING_STOP = False        # exit via trailing stop rather than flat 3-month hold
TIME_EXIT_DAYS = 0               # >0 = cut a name that hasn't recovered pre-drop by day N


def has_director_buy(director_buy) -> bool:
    return director_buy is True or str(director_buy).strip().lower() == "true"


def is_quality(npat_margin, fcf_margin) -> bool:
    """Profitability gate. None (fundamentals missing) fails CLOSED — if we can't
    prove a name isn't deeply loss-making, we don't trade it. Returns True only when
    both margins are known and above their floors."""
    if npat_margin is None or fcf_margin is None:
        return False
    try:
        return float(npat_margin) >= NPAT_MARGIN_MIN and float(fcf_margin) >= FCF_MARGIN_MIN
    except (TypeError, ValueError):
        return False

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
        # LIVE stays S&P 1500 until the Nasdaq widening is validated in the backtest;
        # the backtest overrides this to "us_expanded" (S&P 1500 + Nasdaq). Flip this
        # to "us_expanded" once the re-run confirms the added cohort earns its place.
        "universe": "sp1500",
        "announcements": "sec",
    },
    # Survivorship-free EODHD backtest markets (broadening the book). No SEC insider
    # data outside the US, so these run the GATED model (no director-buy filter) until
    # a per-market director source is wired. Liquidity floors in local currency.
    "TSX": {
        "suffix": ".TO",
        "currency": "CAD",
        "min_market_cap": 200_000_000,       # >= C$200m
        "min_avg_daily_value": 1_000_000,    # C$/day liquidity floor
        "large_cap_cutoff": 10_000_000_000,  # >= C$10bn -> composite benchmark
        "benchmark_large": "^GSPTSE",        # S&P/TSX Composite
        "benchmark_small": "^GSPTSE",        # (no reliable free TSX small-cap index; composite proxy)
        "universe": "eodhd",
        "announcements": "none",
    },
    "LSE": {
        "suffix": ".L",
        "currency": "GBP",
        "min_market_cap": 100_000_000,       # >= £100m
        "min_avg_daily_value": 500_000,      # £/day liquidity floor
        "large_cap_cutoff": 5_000_000_000,   # >= £5bn -> FTSE 100
        "benchmark_large": "^FTSE",          # FTSE 100
        "benchmark_small": "^FTMC",          # FTSE 250 (mid/small proxy for overreaction cohort)
        "universe": "eodhd",
        "announcements": "none",
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

_GLOBALS = ["Z_THRESHOLD", "INDEX_REL_THRESHOLD", "ABS_DROP_THRESHOLD", "MAX_DROP_FLOOR",
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
