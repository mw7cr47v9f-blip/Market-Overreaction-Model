"""
Re-classify director open-market buying at MULTIPLE lookback windows in one pass.

Motivation: the locked model uses a 6-month (182d) director-buy window. Directors
sit in blackout for long stretches (often right before the earnings that cause the
drop), so a 6-month window can miss a real conviction buy purely on timing. This
tests whether widening the runway to 12 / 18 / 24 months recovers those cases —
without deleting the conviction signal, which the data shows is the model's alpha
engine.

Mechanics: fetch each candidate's SEC Form-4 bulk history ONCE, deep enough for the
widest window (24 months), then re-run classify_window at 182 / 365 / 547 / 730 days
off the same filings. Emits director_buy_val_6m / _12m / _18m / _24m columns.

SEC data only — FREE, spends ZERO EODHD credit. Runs in GitHub Actions (SEC is not
reachable from the analysis box). Point-in-time integrity is preserved by
classify_window (a filing dated after the event is ignored).

    python -m screener.reclass_insider --events data/candidates_nodir.csv \
        --out data/candidates_nodir_windows.csv
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import timedelta

from . import us_insiders as ui

# window label -> lookback days (~30.4 d/month; 24m uses 730 to match a clean 2y)
WINDOWS = {"6m": 182, "12m": 365, "18m": 547, "24m": 730}
MAX_WIN = max(WINDOWS.values())


def log(m):
    print(f"[reclass] {m}", file=sys.stderr, flush=True)


def enrich(events_csv: str, out_csv: str, limit: int = 0):
    import pandas as pd
    import requests

    df = pd.read_csv(events_csv, low_memory=False)
    if limit:
        df = df.head(limit)
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = df["date"].astype(str)

    events_by_ticker: dict[str, list[str]] = defaultdict(list)
    for _, r in df.iterrows():
        events_by_ticker[r["ticker"]].append(r["date"][:10])
    log(f"{len(df)} events across {len(events_by_ticker)} tickers; "
        f"windows = {list(WINDOWS)}")

    # --- CIK mapping ---
    try:
        ui._load_cik_map()
    except Exception as e:  # noqa: BLE001
        log(f"CIK map load FAILED: {e!r} — aborting"); raise
    want_ciks: dict[int, str] = {}
    for tk in events_by_ticker:
        cik = ui._cik_for(tk)
        if cik is not None:
            want_ciks[cik] = tk
    log(f"{len(want_ciks)}/{len(events_by_ticker)} tickers mapped to CIK")

    # --- fetch Form-4 quarters, DEEP enough for the widest window ---
    all_dates = [d for evs in events_by_ticker.values() for e in evs
                 if (d := ui._to_date(e)) is not None]
    lo = min(all_dates) - timedelta(days=MAX_WIN + 15)   # << key: 24m of runway
    hi = max(all_dates)
    qs = ui.quarters_between(lo, hi)
    log(f"scanning {len(qs)} quarters {qs[0]}..{qs[-1]} for a {MAX_WIN}d max window")

    session = requests.Session()
    filings_by_ticker: dict[str, list] = defaultdict(list)
    for j, (y, q) in enumerate(qs):
        t0 = time.time()
        z = ui._download_quarter(y, q, session)
        if z is None:
            continue
        idx = ui._index_quarter(z, want_ciks)
        for tk, lst in idx.items():
            filings_by_ticker[tk].extend(lst)
        log(f"  {y}Q{q} -> {sum(len(v) for v in idx.values())} filings "
            f"({j+1}/{len(qs)}, {time.time()-t0:.0f}s)")

    n_filings = sum(len(v) for v in filings_by_ticker.values())
    log(f"fetched {n_filings} filings for {len(filings_by_ticker)} names")

    # --- reclassify each event at every window off the SAME filings ---
    out_cols = {w: [] for w in WINDOWS}
    dir_any = {w: 0 for w in WINDOWS}
    for _, r in df.iterrows():
        filings = filings_by_ticker.get(r["ticker"], [])
        ev = r["date"][:10]
        for w, days in WINDOWS.items():
            c = ui.classify_window(filings, ev, lookback_days=days)
            v = c["director_buy_val"]
            out_cols[w].append(v)
            if v and v >= 50000:
                dir_any[w] += 1

    for w in WINDOWS:
        df[f"director_buy_val_{w}"] = out_cols[w]

    df.to_csv(out_csv, index=False)
    log(f"wrote {out_csv}")
    log("candidates clearing the $50k director-buy floor by window:")
    for w in WINDOWS:
        log(f"    {w:4s}: {dir_any[w]:4d} / {len(df)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/candidates_nodir.csv")
    ap.add_argument("--out", default="data/candidates_nodir_windows.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    enrich(a.events, a.out, a.limit)


if __name__ == "__main__":
    main()
