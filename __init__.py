"""
Configuration for the ASX overreaction screen.

Every threshold below is a deliberate choice flagged in the README. Adjust here
and the whole pipeline follows. Keegan: these are the knobs to turn if the feed
is too noisy (tighten) or too quiet (loosen).
"""

# ---- Step 1: statistical screen thresholds -------------------------------

# Condition 1 — statistical extremity.
# Window return must be at least this many standard deviations below zero,
# where sigma is the stock's own trailing daily volatility scaled to the window
# length. Spec says 2.5-3.0. We default to 2.5 (more inclusive); tighten to 3.0
# to cut the list down.
Z_THRESHOLD = -2.5

# Condition 2 — index-relative underperformance.
# Window return must underperform the benchmark index over the SAME window by at
# least this many percentage points (0.10 = 10pp). Excludes broad selloffs.
INDEX_REL_THRESHOLD = -0.10

# Condition 3 — absolute floor.
# Raw window decline must be at least this large (0.10 = 10%). Keeps out
# statistically-extreme-but-trivial moves in very quiet stocks. This floor is
# also WHY we never need to scan the whole 2000-stock universe for the drop
# itself — anything that qualifies is, by definition, a large faller.
ABS_DROP_THRESHOLD = -0.10

# Rolling windows to evaluate, in trading days. Spec: "5 trading days or less".
# We test every window length 1..5 ending on the run date and flag if ANY of
# them qualifies, reporting the most statistically extreme one.
WINDOW_LENGTHS = [1, 2, 3, 4, 5]

# Trailing sample for the volatility baseline, in trading days (~90).
# Critically, the volatility is computed from returns STRICTLY BEFORE the
# evaluation window, so the crash itself doesn't inflate its own baseline.
VOL_LOOKBACK = 90

# Minimum number of trailing daily returns required to compute a trustworthy
# volatility. Stocks with less history (recent listings) are skipped and logged.
MIN_VOL_OBS = 40

# ---- Universe / size / liquidity filters ---------------------------------

# Minimum market capitalisation in AUD. Spec: >= $100m.
MIN_MARKET_CAP = 100_000_000

# Liquidity floor: minimum average daily traded value (close * volume) over the
# trailing VOL_LOOKBACK window, in AUD. Spec suggested $50k-$100k or bottom
# decile. We use a fixed, slightly conservative $150k/day so a flagged move
# can't be an artefact of a handful of thin trades. Flagged in README.
MIN_AVG_DAILY_VALUE = 150_000

# ---- Benchmarks ----------------------------------------------------------

# Yahoo tickers for the ASX price indices.
#   ^AXJO = S&P/ASX 200, ^AXKO = S&P/ASX 300
# Large caps (market cap >= LARGE_CAP_CUTOFF) are benchmarked against the ASX
# 200; everything else against the ASX 300, per the spec's "whichever is more
# appropriate for its size".
BENCHMARK_200 = "^AXJO"
BENCHMARK_300 = "^AXKO"
LARGE_CAP_CUTOFF = 5_000_000_000  # $5bn: above this, use ASX 200

# ---- Data window ---------------------------------------------------------

# Calendar days of history to download so we have >= VOL_LOOKBACK + 5 trading
# days after weekends/holidays. 200 calendar days comfortably covers ~135
# trading days.
HISTORY_CALENDAR_DAYS = 200

# ---- Currency ------------------------------------------------------------
# yfinance returns ASX prices in AUD already; kept explicit for clarity.
CURRENCY = "AUD"
