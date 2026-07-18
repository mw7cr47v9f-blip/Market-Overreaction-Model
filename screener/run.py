"""
Orchestrator. Runs daily on GitHub Actions after the ASX close.

  python -m screener.run --data-dir data
  python -m screener.run --self-test        # offline logic check, no network

Two-stage flow for efficiency:
  Stage 1  bulk price screen over the WHOLE universe (prices are cheap) ->
           small shortlist that meets z-score / index-relative / >=10% / liquidity
  Stage 2  fetch market cap only for the shortlist, apply the >=$100m size gate
           and the size-appropriate benchmark, finalise figures
Then dedup against state.json and emit only genuinely-new events.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import pandas as pd

from . import config as cfg
from . import data as datamod
from . import state as statemod
from .stats import evaluate_series


def log(msg: str):
    print(f"[run] {msg}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap universe size (debug)")
    args = ap.parse_args()

    if args.self_test:
        from . import test_stats  # noqa: F401  (runs assertions on import)
        return

    os.makedirs(args.data_dir, exist_ok=True)
    state_path = os.path.join(args.data_dir, "state.json")
    all_csv = os.path.join(args.data_dir, "candidates_all.csv")
    universe_fallback = os.path.join(args.data_dir, "universe.csv")

    # 1. Universe
    uni = datamod.get_universe(local_fallback=universe_fallback if os.path.exists(universe_fallback) else None)
    if args.limit:
        uni = uni.head(args.limit)
    name_by_yahoo = dict(zip(uni["yahoo"], uni["name"]))
    code_by_yahoo = dict(zip(uni["yahoo"], uni["code"]))

    # 2. Prices + benchmarks
    prices = datamod.download_prices(list(uni["yahoo"]))
    benches = datamod.download_benchmarks()
    bench300 = benches.get(cfg.BENCHMARK_300)
    bench200 = benches.get(cfg.BENCHMARK_200)
    if bench300 is None and bench200 is None:
        raise RuntimeError("No benchmark index data — cannot compute index-relative move.")
    bench300 = bench300 if bench300 is not None else bench200
    bench200 = bench200 if bench200 is not None else bench300

    # 3. Stage 1 — price screen over everything (size gate bypassed via inf cap)
    shortlist = []
    for y, df in prices.items():
        c = evaluate_series(code_by_yahoo.get(y, y), name_by_yahoo.get(y, y),
                            df["Close"], df["Volume"], bench300,
                            market_cap=math.inf, cfg=cfg)
        if c is not None:
            shortlist.append((y, df))
    log(f"stage 1 price-qualifying shortlist: {len(shortlist)}")

    # 4. Stage 2 — real market cap + size gate + size-appropriate benchmark
    caps = datamod.get_market_caps([y for y, _ in shortlist]) if shortlist else {}
    finals = []
    for y, df in shortlist:
        cap = caps.get(y)
        if cap is None or cap < cfg.MIN_MARKET_CAP:
            continue
        bench = bench200 if cap >= cfg.LARGE_CAP_CUTOFF else bench300
        c = evaluate_series(code_by_yahoo.get(y, y), name_by_yahoo.get(y, y),
                            df["Close"], df["Volume"], bench, market_cap=cap, cfg=cfg)
        if c is not None:
            finals.append(c)
    log(f"stage 2 size-qualifying candidates: {len(finals)}")

    # 5. Dedup against state
    seen = statemod.load_seen(state_path)
    new = [c for c in finals if c.key() not in seen]
    scan_date = _latest_scan_date(prices)
    log(f"genuinely-new events this run: {len(new)} (of {len(finals)} live)")

    # 6. Persist
    new_rows = [c.to_dict() for c in new]
    statemod.append_candidates(all_csv, new_rows)
    statemod.write_new(os.path.join(args.data_dir, "candidates_new.json"), new_rows, scan_date)
    statemod.save_state(state_path, seen.union({c.key() for c in finals}), scan_date)

    # 7. Announcements for the new names (best effort; feeds the analysis step)
    try:
        from . import announcements
        announcements.fetch_for_candidates(new, args.data_dir)
    except Exception as e:  # noqa: BLE001
        log(f"announcement fetch skipped/failed: {e!r}")

    # 8. Human-readable console summary (also captured in the Actions log)
    _print_summary(new, scan_date)


def _latest_scan_date(prices: dict) -> str:
    latest = None
    for df in prices.values():
        d = df.index.max()
        if latest is None or d > latest:
            latest = d
    return pd.Timestamp(latest).date().isoformat() if latest is not None else "unknown"


def _print_summary(new, scan_date):
    print("\n" + "=" * 68)
    print(f"ASX OVERREACTION SCREEN — scan date {scan_date}")
    print(f"New candidates: {len(new)}")
    for c in sorted(new, key=lambda x: x.z_score):
        print(f"  {c.ticker:6} {c.name[:28]:28} "
              f"{c.raw_return*100:6.1f}%  idx-rel {c.index_relative*100:6.1f}pp  "
              f"z={c.z_score:5.1f}  {c.window_len}d  cap ${c.market_cap/1e6:,.0f}m")
    print("=" * 68)


if __name__ == "__main__":
    main()
