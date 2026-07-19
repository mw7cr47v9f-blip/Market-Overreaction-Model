"""
Orchestrator. Runs daily on GitHub Actions, screening every configured market
in one pass.

  python -m screener.run --data-dir data
  python -m screener.run --markets ASX          # limit to one market
  python -m screener.run --self-test            # offline logic check, no network

Per market, two-stage flow: bulk price screen over the whole universe ->
small shortlist meeting z / index-relative / >=10% / liquidity -> fetch market
cap for the shortlist, apply the size gate + size-appropriate benchmark ->
finalise. Then dedup all markets against state.json and emit only new events.

Timing note: one daily run screens both markets on their latest COMPLETED
session. At ~08:00 UTC that's the same-day ASX close and the prior US close
(the most recent finished US session) — nothing is missed; US just lands about
half a day after its close. Add a second post-US-close schedule later if you
want US caught fresher.
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
from . import ledger as ledgermod
from .stats import evaluate_series


def log(msg: str):
    print(f"[run] {msg}", file=sys.stderr, flush=True)


def screen_market(name: str, data_dir: str, limit: int = 0):
    """Return the list of live (size-qualifying) Candidates for one market."""
    mcfg = cfg.market_params(name)
    log(f"=== market {name} ===")
    fallback = os.path.join(data_dir, f"universe_{name}.csv")
    uni = datamod.get_universe(mcfg, local_fallback=fallback if os.path.exists(fallback) else None)
    if limit:
        uni = uni.head(limit)
    name_by_y = dict(zip(uni["yahoo"], uni["name"]))
    code_by_y = dict(zip(uni["yahoo"], uni["code"]))

    prices = datamod.download_prices(list(uni["yahoo"]))
    benches = datamod.download_benchmarks(mcfg)
    b_small = benches.get(mcfg.BENCHMARK_300)   # None or a pandas Series
    b_large = benches.get(mcfg.BENCHMARK_200)
    if b_small is None and b_large is None:
        log(f"{name}: no benchmark data — skipping market")
        return [], prices, uni
    if b_small is None:
        b_small = b_large
    if b_large is None:
        b_large = b_small

    # Stage 1 — price screen over everything (size gate bypassed via inf cap)
    shortlist = []
    for y, df in prices.items():
        c = evaluate_series(code_by_y.get(y, y), name_by_y.get(y, y),
                            df["Close"], df["Volume"], b_small, market_cap=math.inf, cfg=mcfg)
        if c is not None:
            shortlist.append((y, df))
    log(f"{name}: stage-1 price-qualifying = {len(shortlist)}")

    # Stage 2 — real cap + size gate + size-appropriate benchmark
    caps = datamod.get_market_caps([y for y, _ in shortlist]) if shortlist else {}
    finals = []
    for y, df in shortlist:
        cap = caps.get(y)
        if cap is None or cap < mcfg.MIN_MARKET_CAP:
            continue
        bench = b_large if cap >= mcfg.LARGE_CAP_CUTOFF else b_small
        c = evaluate_series(code_by_y.get(y, y), name_by_y.get(y, y),
                            df["Close"], df["Volume"], bench, market_cap=cap, cfg=mcfg)
        if c is not None:
            finals.append(c)
    log(f"{name}: stage-2 size-qualifying = {len(finals)}")
    return finals, prices, uni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--markets", default="", help="comma list, e.g. ASX,US (default: all)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap universe size (debug)")
    args = ap.parse_args()

    if args.self_test:
        from . import test_stats   # noqa: F401  (runs assertions on import)
        from . import test_ledger  # noqa: F401
        return

    os.makedirs(args.data_dir, exist_ok=True)
    state_path = os.path.join(args.data_dir, "state.json")
    all_csv = os.path.join(args.data_dir, "candidates_all.csv")

    markets = [m.strip() for m in args.markets.split(",") if m.strip()] or list(cfg.MARKETS)

    all_finals, latest_by_market, all_prices = [], {}, {}
    sector_by_mc, yahoo_by_mc = {}, {}   # (market, code) -> sector / yahoo ticker
    for m in markets:
        try:
            finals, prices, uni = screen_market(m, args.data_dir, args.limit)
            all_finals.extend(finals)
            all_prices.update(prices)
            latest_by_market[m] = _latest_scan_date(prices)
            for _, row in uni.iterrows():
                sector_by_mc[(m, row["code"])] = row.get("sector")
                yahoo_by_mc[(m, row["code"])] = row["yahoo"]
        except Exception as e:  # noqa: BLE001
            log(f"market {m} failed: {e!r}")

    def price_lookup(market, ticker):
        y = yahoo_by_mc.get((market, str(ticker).upper()))
        df = all_prices.get(y) if y else None
        if df is None or len(df) == 0:
            return None
        s = df["Close"].dropna()
        return float(s.iloc[-1]) if len(s) else None

    # Dedup across all markets
    seen = statemod.load_seen(state_path)
    new = [c for c in all_finals if c.key() not in seen]
    scan_date = _latest_scan_date(all_prices)
    log(f"new events this run: {len(new)} (of {len(all_finals)} live across {len(markets)} market(s))")

    # Attach sector + the locked-model favoured/avoid tags to every new candidate.
    new_rows = []
    for c in new:
        d = c.to_dict()
        sec = sector_by_mc.get((c.market, c.ticker))
        d["sector"] = sec
        d["favoured"] = cfg.is_favoured(sec)
        d["avoided"] = cfg.is_avoided(sec)
        new_rows.append(d)
    favoured_new = [d for d in new_rows if d["favoured"]]
    log(f"favoured-sector (alerted): {len(favoured_new)} of {len(new_rows)} new")

    # Log EVERYTHING (with tags) to the running CSV; ALERT only favoured names.
    statemod.append_candidates(all_csv, new_rows)
    statemod.write_new(os.path.join(args.data_dir, "candidates_new.json"), favoured_new, scan_date)
    statemod.save_state(state_path, seen.union({c.key() for c in all_finals}), scan_date)

    # BUY/SELL ledger: model book (favoured signals, 3-month hold) + personal holdings.
    try:
        model_buys = [{"market": d["market"], "ticker": d["ticker"], "name": d["name"],
                       "sector": d.get("sector"), "entry_date": scan_date,
                       "entry_price": d["last_close"]} for d in favoured_new]
        model_report = ledgermod.update_model_ledger(args.data_dir, model_buys, price_lookup, scan_date)
        holdings = ledgermod.holdings_status(args.data_dir, price_lookup, scan_date)
        ledgermod.write_status(args.data_dir, model_report, holdings)
        log(f"ledger: {len(model_report['new_buys'])} new, {model_report['n_open']} open, "
            f"{len(model_report['closed_this_run'])} closed this run, {len(holdings)} personal holdings")
    except Exception as e:  # noqa: BLE001
        log(f"ledger update failed: {e!r}")

    # Primary-source pre-fetch, dispatched per market (favoured names only)
    favoured_keys = {(d["market"], d["ticker"], d["event_date"]) for d in favoured_new}
    _fetch_announcements([c for c in new if (c.market, c.ticker, c.event_date) in favoured_keys],
                         args.data_dir)
    _print_summary(new, scan_date, latest_by_market)


def _fetch_announcements(new, data_dir):
    by_src = {}
    for c in new:
        src = cfg.MARKETS.get(c.market, {}).get("announcements")
        by_src.setdefault(src, []).append(c)
    if by_src.get("asx"):
        try:
            from . import announcements
            announcements.fetch_for_candidates(by_src["asx"], data_dir)
        except Exception as e:  # noqa: BLE001
            log(f"ASX announcement fetch failed: {e!r}")
    if by_src.get("sec"):
        try:
            from . import us_announcements
            us_announcements.fetch_for_candidates(by_src["sec"], data_dir)
        except Exception as e:  # noqa: BLE001
            log(f"SEC filing fetch failed: {e!r}")


def _latest_scan_date(prices: dict) -> str:
    latest = None
    for df in prices.values():
        d = df.index.max()
        if latest is None or d > latest:
            latest = d
    return pd.Timestamp(latest).date().isoformat() if latest is not None else "unknown"


def _print_summary(new, scan_date, latest_by_market):
    print("\n" + "=" * 72)
    print(f"OVERREACTION SCREEN — scan date {scan_date}")
    for m, d in latest_by_market.items():
        print(f"  {m}: latest session {d}")
    print(f"New candidates: {len(new)}")
    for c in sorted(new, key=lambda x: x.z_score):
        sym = "$" if c.currency == "USD" else "A$"
        print(f"  [{c.market:3}] {c.ticker:6} {c.name[:26]:26} "
              f"{c.raw_return*100:6.1f}%  idx-rel {c.index_relative*100:6.1f}pp  "
              f"z={c.z_score:5.1f}  {c.window_len}d  cap {sym}{c.market_cap/1e6:,.0f}m")
    print("=" * 72)


if __name__ == "__main__":
    main()
