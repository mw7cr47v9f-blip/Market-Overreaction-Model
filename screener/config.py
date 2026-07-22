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
ABS_DROP_THRESHOLD = -0.20   # raw decline >= 20%. The credit-line study showed return
                             # scales hard with drop depth: 15-20% falls recover ~+10%,
                             # 20-30% ~+16%, 30-60% ~+30%. Raising the floor to 20% roughly
                             # doubles the return on a fixed facility (fewer, deeper, higher-
                             # win trades); the multi-market build offsets the lower count.
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

# ---- Sector scope (keep everything EXCEPT the avoid-list) --------------------
# UPDATED from the full survivorship-free NYSE run: with the director + value gates
# doing the selecting, restricting to FAVOURED sectors threw away good trades.
# Broadening to "all-except-avoid" lifted the book 92 -> 198 trades AND raised the
# per-trade return (+22.7% -> +24.2%) and hit rate (77% -> 80%). So we keep the
# avoid-list (it still earns its place) but drop the favoured-only wall.
import re as _re
# "technology" (bare) covers S&P "Information Technology", ASX "Technology Hardware
# & Equipment" AND Nasdaq's plain "Technology" label — the three feeds name it
# differently, so match the common root.
FAVOURED_SECTORS_RE = (r"\btechnology|software|semiconductor|"
                       r"discretionary|consumer cyclical|industrial|capital goods|automobile")
# Note: EODHD tags consumer-discretionary names as "Consumer Cyclical" (not the S&P
# "Consumer Discretionary"), so both labels are matched. "consumer cyclical" is written
# out in full so it does NOT match "Consumer Non-Cyclicals" (staples).
# Energy: tested in and out as risk-reduction analysis. It carried favoured-level per-trade
# excess (+3.3%) but its year-by-year contribution was a coin toss — including it, the book
# beat the S&P calendar return in only 9 of 15 years (60%); excluding it, 11 of 15 (73%),
# with a higher average annual return on deployed capital. Removed. Consumer Defensive was
# also tested and left out (raised volatility without adding return).
AVOID_SECTORS_RE = r"material|pharma|biotech|telecom|communication|real estate"
# "communication" added: Communication Services was the worst sector (46% hit, -6.9%
# excess, negative Sharpe) — an explicit hard-avoid.
FAVOURED_ONLY = False      # False = broad (keep all EXCEPT avoids, the locked model);
                           # True  = revert to the narrow favoured-only book.
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


def is_sector_ok(sector) -> bool:
    """The live sector gate. Broad by default: keep everything EXCEPT the avoid-list.
    Set FAVOURED_ONLY=True to revert to the narrow favoured-only book."""
    if is_avoided(sector):
        return False
    return is_favoured(sector) if FAVOURED_ONLY else True


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
REQUIRE_DIRECTOR_BUY = True      # LIVE now enforces the locked-model director-buy hard filter
USE_TRAILING_STOP = False        # exit via trailing stop rather than flat 3-month hold
TIME_EXIT_DAYS = 0               # >0 = cut a name that hasn't recovered pre-drop by day N
# Size floor on the director buy (the "wife/husband test"): the full NYSE run showed
# sub-$50k "goodwill" buys hit only ~60% — no better than no buy at all — while
# reliability climbs with size ($250k-1m ~72%, $1m+ ~75%). A $50k floor drops the noise
# at essentially no cost to return (198 -> 174 trades, same +24% / 80%). Set to 0 to
# count any director buy.
DIRECTOR_BUY_MIN_VAL = 50000


def has_director_buy(director_buy) -> bool:
    return director_buy is True or str(director_buy).strip().lower() == "true"


def director_buy_ok(director_buy, director_buy_val=None) -> bool:
    """Director-buy hard filter WITH the size floor. Counts only if a director bought AND
    the $ size clears DIRECTOR_BUY_MIN_VAL. Fails OPEN on unknown size (None) so a missing
    value never silently drops a genuine buy."""
    if not has_director_buy(director_buy):
        return False
    if director_buy_val is None or not DIRECTOR_BUY_MIN_VAL:
        return True
    try:
        return float(director_buy_val) >= DIRECTOR_BUY_MIN_VAL
    except (TypeError, ValueError):
        return True


# ---- Valuation gate (from the P/E-vs-sector recovery study) ------------------
# Oversold names that were EXPENSIVE vs their sector BEFORE the drop recover materially
# worse: in the gated director book the cheapest third hit 68% / +14.3%, the most
# expensive third only 58% / +8.2%. The effect is monotonic, holds in 13 of 15 years
# and within every drop-size bucket, is uncorrelated with the margin gate, and stays
# significant (p=0.0008) after controlling for year, drop size AND margins — a value
# effect (overvalued names partly deserve the drop). So we HARD-EXCLUDE names whose
# pre-drop P/E was more than VALUE_REL_PE_MAX x their sector's median. Fails OPEN: a
# loss-maker or a name with no usable P/E is KEPT (only positively-expensive names go).
REQUIRE_VALUE = True
VALUE_REL_PE_MAX = 1.54            # exclude pre-drop P/E > 1.54x the sector median
_SECTOR_PE_MEDIAN = {             # median pre-drop P/E by favoured sector (US backtest)
    "technology": 28.5, "software": 28.5, "semiconductor": 28.5,
    "discretionary": 15.0, "consumer cyclical": 15.0, "automobile": 15.0,
    "industrial": 18.7, "capital goods": 18.7,
}


def _sector_pe_median(sector):
    s = str(sector).lower()
    for k, v in _SECTOR_PE_MEDIAN.items():
        if k in s:
            return v
    return None


def predrop_pe(earnings_yield, raw):
    """Reconstruct the PRE-drop P/E from the post-drop earnings yield and the drop
    return `raw` (negative). Returns None when not usable (loss-maker / data tail)."""
    try:
        ey = float(earnings_yield)
        r = float(raw)
    except (TypeError, ValueError):
        return None
    ey_pre = ey * (1 + r)
    if ey_pre <= 0:
        return None
    pe = 1.0 / ey_pre
    return pe if 3 <= pe <= 150 else None


def is_value_ok(earnings_yield, raw, sector) -> bool:
    """Valuation gate. Fails OPEN — a name with no usable P/E or no sector reference is
    KEPT (True); only a name positively identified as expensive vs its sector is dropped."""
    if not REQUIRE_VALUE:
        return True
    pe = predrop_pe(earnings_yield, raw)
    if pe is None:
        return True
    m = _sector_pe_median(sector)
    if not m:
        return True
    return pe <= VALUE_REL_PE_MAX * m


# ---- Conviction-weighted position sizing (credit-line study) -----------------
# On a fixed facility, return = per-trade return x turnover, and the return lives in the
# deepest, cheapest drops. Rather than equal-weight, size each stake by conviction — drop
# depth x cheapness-vs-sector — which lifted the capital-weighted return ~+2.3pts (13.5%
# -> 15.8%) with NO change to the trade set. The screen emits this weight; size the stake
# relative to it. Weight ~1.0 is a normal position; the clips bound concentration.
USE_CONVICTION_WEIGHT = True


def conviction_weight(raw, earnings_yield, sector) -> float:
    """Relative position size for a confirmed buy. Deeper drops and cheaper-vs-sector
    names get a bigger stake; bounded so no single trade dominates the facility. Missing
    valuation -> cheapness factor 1.0 (neutral). Returns a multiplier centred near 1."""
    if not USE_CONVICTION_WEIGHT:
        return 1.0
    try:
        drop = abs(float(raw))
    except (TypeError, ValueError):
        return 1.0
    depth = min(max(drop / 0.20, 0.5), 2.5)          # normalise to the 20% floor
    cheap = 1.0
    pe = predrop_pe(earnings_yield, raw)
    if pe is not None:
        m = _sector_pe_median(sector)
        if m:
            cheap = min(max(1.5 - pe / m, 0.5), 2.0)  # cheaper than sector -> bigger
    return round(depth * cheap, 3)


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
        # LIVE universe is NYSE-listed (the backtest's scope), sourced from a free
        # GitHub mirror carrying sector + marketCap — so live matches the backtest and
        # needs NO EODHD key. The >=US$1bn floor bounds the daily yfinance pull; the
        # screen's stage-2 gate re-verifies caps. NASDAQ is a tested-but-not-yet-adopted
        # widening (would change "universe" to "us_expanded").
        "min_market_cap": 1_000_000_000,     # >= US$1bn
        "min_avg_daily_value": 2_000_000,    # US$/day liquidity floor (US trades heavier)
        "large_cap_cutoff": 20_000_000_000,  # >= US$20bn benchmarks vs S&P 500
        "benchmark_large": "^GSPC",          # S&P 500
        "benchmark_small": "^RUT",           # Russell 2000
        "universe": "nyse",
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
    # Asian expansion (backtest only for now) — no SEC-style insider data, so they run the
    # gated model (director applied live by analogy). Added to fatten the trade count that
    # the 20% drop floor thins. EODHD exchange codes: TSE / SG / HK.
    "TSE": {                                 # Tokyo (Japan)
        "suffix": ".T",
        "currency": "JPY",
        "min_market_cap": 20_000_000_000,    # >= ¥20bn (~US$130m)
        "min_avg_daily_value": 50_000_000,   # ¥/day liquidity floor
        "large_cap_cutoff": 1_000_000_000_000,  # >= ¥1tn -> Nikkei
        "benchmark_large": "^N225",          # Nikkei 225
        "benchmark_small": "^N225",          # (no reliable free JP small-cap index; proxy)
        "universe": "eodhd",
        "announcements": "none",
    },
    "SGX": {                                 # Singapore
        "suffix": ".SI",
        "currency": "SGD",
        "min_market_cap": 200_000_000,       # >= S$200m
        "min_avg_daily_value": 300_000,      # S$/day liquidity floor
        "large_cap_cutoff": 5_000_000_000,   # >= S$5bn -> STI
        "benchmark_large": "^STI",           # Straits Times Index
        "benchmark_small": "^STI",
        "universe": "eodhd",
        "announcements": "none",
    },
    "SEHK": {                                # Hong Kong
        "suffix": ".HK",
        "currency": "HKD",
        "min_market_cap": 1_000_000_000,     # >= HK$1bn (~US$130m)
        "min_avg_daily_value": 3_000_000,    # HK$/day liquidity floor
        "large_cap_cutoff": 40_000_000_000,  # >= HK$40bn -> Hang Seng
        "benchmark_large": "^HSI",           # Hang Seng
        "benchmark_small": "^HSI",
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
