"""
Back-fill a `market_cap` column onto an EXISTING backtest events file, WITHOUT
re-running the whole backtest. The original runs already qualified these events on
z / index-relative / liquidity / collapse-floor; they just never wrote market cap.

Cost control: only the CANDIDATE POOL is priced — events that already pass the cheap,
always-on gates (drop ≥ floor, quality, sector). For each unique ticker in that pool
we make exactly ONE fundamentals call (shares outstanding) and ONE EOD call (its whole
event-date range), then look up each event's close. So ~2 calls per ticker, not per
event. market_cap = shares_out x adjusted_close_at_event.

Runs on GitHub Actions with EODHD_API_TOKEN (SEC not needed here). Aborts cleanly on a
402 (daily credit limit) and writes what it has, so a quota-hit never loses progress.

    python -m screener.backfill_marketcap --events data/backtest_events_US.csv \
        --out data/backtest_events_US_cap.csv --drop-floor 0.20

Note: shares_out is the latest reported count (a coarse proxy for a $1bn floor); for
delisted names that is the count near delisting, close enough for a size cut. The
native emit path in backtest_factor.py is the precise, point-in-time version.
"""
from __future__ import annotations

import argparse
import sys

from . import config as cfg
from . import eodhd


def log(m):
    print(f"[backfill-cap] {m}", file=sys.stderr, flush=True)


def _candidate_pool(df, drop_floor):
    """The events worth pricing: pass the cheap always-on gates so we don't spend
    credits on names that can never be a trade regardless of cap."""
    import pandas as pd
    raw = pd.to_numeric(df["raw"], errors="coerce")
    npat = pd.to_numeric(df.get("npat_margin"), errors="coerce")
    fcf = pd.to_numeric(df.get("fcf_margin"), errors="coerce")
    m = (raw <= -abs(drop_floor)) & (raw >= cfg.MAX_DROP_FLOOR)
    m &= (npat >= cfg.NPAT_MARGIN_MIN) & (fcf >= cfg.FCF_MARGIN_MIN)
    m &= ~df["sector"].astype(str).str.lower().map(cfg.is_avoided)
    return df[m]


def backfill(events_csv, out_csv, drop_floor=0.20):
    import pandas as pd
    import requests
    df = pd.read_csv(events_csv, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    pool = _candidate_pool(df, drop_floor).copy()
    tickers = sorted(pool["ticker"].dropna().unique().tolist())
    log(f"{len(df)} events -> {len(pool)} in candidate pool across {len(tickers)} tickers "
        f"(drop >= {drop_floor:.0%}, quality, sector)")

    sess = requests.Session()
    cap_by_key = {}          # (ticker, date) -> market_cap
    done = aborted = 0
    for tk in tickers:
        sub = pool[pool["ticker"] == tk]
        market = str(sub["market"].iloc[0]) if "market" in sub.columns else "US"
        try:
            fund = eodhd.fundamentals(tk, market, session=sess)
            shares = (fund or {}).get("shares_out")
            if not shares:
                continue
            lo = sub["date"].min().strftime("%Y-%m-%d")
            hi = sub["date"].max().strftime("%Y-%m-%d")
            px = eodhd.eod_series(tk, market, lo, hi, session=sess)
            if px is None or "Close" not in px:
                continue
            closes = px["Close"].dropna()
            for _, r in sub.iterrows():
                at = closes[closes.index <= r["date"]]
                if len(at):
                    cap_by_key[(tk, r["date"])] = float(shares) * float(at.iloc[-1])
            done += 1
        except Exception as e:  # noqa: BLE001
            log(f"{tk}: {e!r}")
        # quota guard: if EODHD starts 402-ing, every further call fails -> stop, keep progress
        n402, nok = eodhd.quota_stats()
        if n402 >= 20 and n402 > nok * 4:
            aborted = 1
            log(f"ABORT: EODHD daily credit limit hit ({n402} x 402). Saving partial and stopping.")
            break

    pool["market_cap"] = [cap_by_key.get((r.ticker, r.date)) for r in pool.itertuples()]
    got = pool["market_cap"].notna().sum()
    pool.to_csv(out_csv, index=False)
    log(f"priced {done}/{len(tickers)} tickers; market_cap set on {got}/{len(pool)} pool events.")
    log(f"wrote {out_csv}" + ("  [PARTIAL — quota abort; re-run after 00:00 UTC reset to finish]" if aborted else ""))
    return 0 if not aborted else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-floor", type=float, default=0.20,
                    help="price the pool at/below this drop (0.20 leaves room to toggle 20-22%)")
    a = ap.parse_args()
    try:
        raise SystemExit(backfill(a.events, a.out, a.drop_floor))
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log(f"failed: {e!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
