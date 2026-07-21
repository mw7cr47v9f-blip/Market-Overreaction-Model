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
from . import us_insiders
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
        from . import test_stats     # noqa: F401  (runs assertions on import)
        from . import test_ledger    # noqa: F401
        from . import test_universe  # noqa: F401
        from . import test_factor    # noqa: F401  (gate + guards)
        from . import test_sec       # noqa: F401  (live fundamentals)
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
    log(f"favoured-sector: {len(favoured_new)} of {len(new_rows)} new")

    # Profitability gate (survivorship-free factor study): only alert names that are
    # not deeply loss-making. Needs npat_margin + fcf_margin attached per candidate.
    # Fails CLOSED — a name whose fundamentals we can't confirm is NOT alerted.
    if cfg.REQUIRE_QUALITY and favoured_new:
        # Attach point-in-time-ish margins from SEC companyfacts (free) for US names,
        # so the profitability gate can act. Only the day's favoured candidates -> few calls.
        us_fav = [d for d in favoured_new if d.get("market") == "US"]
        if us_fav:
            try:
                from . import sec_fundamentals as secf
                margins = secf.margins_for_tickers([d["ticker"] for d in us_fav])
                for d in us_fav:
                    mm = margins.get(d["ticker"], {})
                    d["npat_margin"] = mm.get("npat_margin")
                    d["fcf_margin"] = mm.get("fcf_margin")
            except Exception as e:  # noqa: BLE001
                log(f"SEC fundamentals fetch failed: {e!r}")

    if cfg.REQUIRE_QUALITY:
        fundamentals_seen = any(("npat_margin" in d) for d in favoured_new)
        if fundamentals_seen:
            # A fundamentals source is wired: gate per-name, failing CLOSED on any
            # name whose margins we can't confirm above the floors.
            for d in favoured_new:
                d["quality_ok"] = cfg.is_quality(d.get("npat_margin"), d.get("fcf_margin"))
            gated = [d for d in favoured_new if d["quality_ok"]]
            log(f"quality-gated (alerted): {len(gated)} of {len(favoured_new)} favoured")
            favoured_new = gated
        elif favoured_new:
            # No fundamentals source yet: don't go dark — pass favoured through but
            # flag loudly that the profitability gate is inactive pending a data feed.
            log("WARNING: quality gate ON but NO fundamentals source wired — alerting "
                f"all {len(favoured_new)} favoured UNGATED. Attach npat_margin/fcf_margin.")
    else:
        log(f"quality gate OFF — alerting all {len(favoured_new)} favoured")

    # Structural-trigger hard filter (8-K study): drop names whose fall was caused by a
    # lost contract / impairment / distress event — those don't mean-revert. Fails OPEN:
    # only a positively-identified structural 8-K excludes a name; 'none'/other are kept.
    if cfg.EXCLUDE_STRUCTURAL_TRIGGERS and favoured_new:
        us_fav = [d for d in favoured_new if d.get("market") == "US"]
        if us_fav:
            try:
                import requests
                from . import sec_triggers as stg
                us_insiders._load_cik_map()          # ensure CIK cache is populated
                sess = requests.Session()
                kept = []
                for d in favoured_new:
                    if d.get("market") != "US":
                        kept.append(d); continue
                    cik = us_insiders._cik_for(str(d["ticker"]).split("_")[0].split("-")[0])
                    rows = stg.fetch_8k_rows(cik, sess) if cik else []
                    trig = stg.triggers_near(rows, d.get("event_date"))
                    d["trigger_primary"] = trig["primary"]
                    d["trigger_cats"] = trig["cats"]
                    if cfg.is_bad_trigger(trig["primary"]):
                        log(f"EXCLUDED {d['ticker']}: structural trigger '{trig['primary']}'")
                    else:
                        kept.append(d)
                log(f"trigger-filtered (alerted): {len(kept)} of {len(favoured_new)} (dropped "
                    f"{len(favoured_new)-len(kept)} structural)")
                favoured_new = kept
            except Exception as e:  # noqa: BLE001
                log(f"trigger filter failed (keeping all): {e!r}")

    # Director-buy hard filter (locked model): keep only US names with prior-6-month
    # on-market DIRECTOR BUYING (SEC Form 4). This is the decisive filter in the
    # survivorship-free backtest. Fails CLOSED per name (no confirmed buy => not
    # alerted); on a total SEC outage it fails OPEN (keeps all, warns) so the feed does
    # not go dark — the daily-analysis agent re-confirms director buys as a backstop.
    # Non-US markets have no SEC Form 4 data, so the filter is applied to US only and
    # non-US names pass through (director rule applied live by analogy).
    if cfg.REQUIRE_DIRECTOR_BUY and favoured_new:
        us_fav = [d for d in favoured_new if d.get("market") == "US"]
        non_us = [d for d in favoured_new if d.get("market") != "US"]
        if us_fav:
            try:
                ebt = {str(d["ticker"]): [scan_date] for d in us_fav}
                sigs = us_insiders.insider_signals_for_events(ebt, years=1)
                kept = []
                for d in us_fav:
                    sig = sigs.get((str(d["ticker"]), scan_date), {}) or {}
                    d["director_buy"] = bool(sig.get("director_buy"))
                    if d["director_buy"]:
                        kept.append(d)
                    else:
                        log(f"EXCLUDED {d['ticker']}: no prior-6-month director buy")
                log(f"director-gated (alerted): {len(kept)} of {len(us_fav)} US favoured")
                favoured_new = kept + non_us
            except Exception as e:  # noqa: BLE001
                log(f"director filter failed (keeping all, UNGATED): {e!r}")

    # Log EVERYTHING (with tags) to the running CSV; ALERT only favoured names.
    statemod.append_candidates(all_csv, new_rows)
    statemod.write_new(os.path.join(args.data_dir, "candidates_new.json"), favoured_new, scan_date)
    statemod.save_state(state_path, seen.union({c.key() for c in all_finals}), scan_date)

    # CONFIRMED-ENTRY tracking (closes the live gap): a gated drop is NOT bought on the
    # fall — it enters a 15-trading-day confirmation window and is BOUGHT only on the
    # confirmed-entry breakout (close above the prior two-day high on above-average
    # volume). Names that never confirm are skipped. Uses the SAME rule as the backtest.
    confirmed_today = []
    try:
        from . import confirm as confirmmod

        def _series_for(rec):
            y = yahoo_by_mc.get((rec.get("market"), str(rec.get("ticker")).upper()))
            df = all_prices.get(y) if y else None
            if df is None or len(df) == 0:
                return None
            return df["Close"], df["Volume"]

        confirmed_today, pending_now, expired_now = confirmmod.update(
            args.data_dir, favoured_new, _series_for, scan_date)
        log(f"confirmations: {len(confirmed_today)} confirmed (BUY today), "
            f"{len(pending_now)} pending, {len(expired_now)} expired/skipped")
    except Exception as e:  # noqa: BLE001
        log(f"confirmation tracking failed: {e!r}")

    # BUY/SELL ledger: the model book buys on CONFIRMED entries (entry date + price are
    # the breakout bar, not the drop), holds three months. Plus personal holdings.
    try:
        model_buys = [{"market": c["market"], "ticker": c["ticker"], "name": c.get("name"),
                       "sector": c.get("sector"),
                       "entry_date": c.get("entry_date", scan_date),
                       "entry_price": c.get("entry_price")} for c in confirmed_today]
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
