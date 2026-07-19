"""
Factor / quality-gate calibration backtest — INDICATIVE (free data).

Purpose: not "did the strategy work" but "which gate criteria actually sort
forward returns among oversold names." Generates every oversold event over
~N years across both markets (the price screen alone, no gate), attaches five
candidate metrics as-of the event, measures 3- and 6-month forward returns
(absolute and excess vs the market), then buckets returns by each metric.

HONEST CAVEATS baked into the output:
  * Universe = today's membership; delisted names excluded (survivorship bias).
  * Fundamentals from yfinance are survivor-only and as-restated, not
    as-first-reported point-in-time, and only ~4y deep. So metric buckets are
    INDICATIVE — enough to see whether the gate direction has signal, not to
    pin a threshold. Upgrade to a point-in-time feed (e.g. EODHD) for that.

Run on GitHub Actions:  python -m screener.backtest_factor --years 5 --data-dir data
Offline logic check:    python -m screener.backtest_factor --self-test
"""
from __future__ import annotations

import argparse, json, math, os, sys
import numpy as np
import pandas as pd

from . import config as cfg
from . import data as datamod

H3, H6 = 63, 126           # ~3 and ~6 trading months
COOLDOWN = 21              # trading days between distinct events for one name
# Backtest keeps a WIDER 10% net than the live screen's 15% floor, so we can still
# bucket by drop size and re-confirm the deeper-is-better finding on future runs.
BT_DROP_FLOOR = -0.10
MOM_LOOKBACK = 126        # 6m pre-drop momentum
US_BOND_TICKER = "^TNX"    # US 10y yield (yahoo)
AU_BOND_YIELD = 0.043     # AU 10y proxy (flagged; no clean free daily series)


def log(m): print(f"[factor] {m}", file=sys.stderr, flush=True)


# ---- vectorised event detection ------------------------------------------

def detect_events(close, volume, bench, mcfg, cooldown=COOLDOWN):
    """Return list of event dicts for one ticker (price screen only)."""
    close = close.dropna().sort_index()
    if len(close) < cfg.VOL_LOOKBACK + max(cfg.WINDOW_LENGTHS) + 5:
        return []
    rets = close.pct_change()
    vol = rets.rolling(cfg.VOL_LOOKBACK).std(ddof=1)
    bench = bench.reindex(close.index).ffill()
    adv = (close * volume.reindex(close.index)).rolling(cfg.VOL_LOOKBACK).mean()

    best_z = pd.Series(np.inf, index=close.index)
    best_w = pd.Series(0, index=close.index)
    qual = pd.Series(False, index=close.index)
    for w in cfg.WINDOW_LENGTHS:
        wret = close / close.shift(w) - 1
        base_vol = vol.shift(w)                      # volatility strictly before the window
        z = wret / (base_vol * math.sqrt(w))
        idxrel = wret - (bench / bench.shift(w) - 1)
        cond = ((z <= cfg.Z_THRESHOLD) & (wret <= BT_DROP_FLOOR) &
                (idxrel <= cfg.INDEX_REL_THRESHOLD) & (adv >= mcfg.MIN_AVG_DAILY_VALUE))
        upd = cond & (z < best_z)
        best_z = best_z.where(~upd, z)
        best_w = best_w.where(~upd, w)
        qual = qual | cond.fillna(False)

    events, last = [], -10**9
    for i in range(len(close)):
        if not bool(qual.iloc[i]) or (i - last) < cooldown:
            continue
        last = i
        w = int(best_w.iloc[i])
        events.append({"pos": i, "date": close.index[i].date().isoformat(),
                       "window_len": w, "z": round(float(best_z.iloc[i]), 3),
                       "raw": round(float(close.iloc[i] / close.iloc[i - w] - 1), 4),
                       "entry": float(close.iloc[i])})
    return events


def forward_and_momentum(close, bench, ev):
    """Attach 3m/6m forward + excess returns and pre-drop momentum to an event."""
    close = close.dropna().sort_index()
    bench = bench.reindex(close.index).ffill()
    i, w, n = ev["pos"], ev["window_len"], len(close)
    out = {}
    for tag, H in (("3m", H3), ("6m", H6)):
        j = i + H
        if j < n:
            fwd = close.iloc[j] / close.iloc[i] - 1
            bfwd = bench.iloc[j] / bench.iloc[i] - 1
            out[f"fwd_{tag}"] = round(float(fwd), 4)
            out[f"excess_{tag}"] = round(float(fwd - bfwd), 4)
        else:
            out[f"fwd_{tag}"] = None
            out[f"excess_{tag}"] = None
    k = i - w
    out["momentum_6m"] = (round(float(close.iloc[k] / close.iloc[k - MOM_LOOKBACK] - 1), 4)
                          if k - MOM_LOOKBACK >= 0 else None)
    return out


# ---- entry timing: buy a CONFIRMED reversal, not the falling close ---------

ENTRY_MAX_WAIT = 15   # trading days to wait for confirmation before skipping


def _sma(series, i, n):
    seg = series.iloc[max(0, i - n):i].dropna()
    return float(seg.mean()) if len(seg) else float("nan")


def find_entry(close, volume, i, rule, max_wait=ENTRY_MAX_WAIT):
    """First confirmed up-day after signal position i, or None (=> skip trade).
    'up1'       — naive: first day that simply closes up.
    'confirmed' — agreed rule: close ABOVE the higher of the prior two closes
                  (a structure break; close-based proxy for 'prior 2-day high'),
                  on volume above its 20-day average. Filters dead-cat bounces."""
    n = len(close)
    for j in range(i + 1, min(i + 1 + max_wait, n)):
        c, c1 = close.iloc[j], close.iloc[j - 1]
        c2 = close.iloc[j - 2] if j >= 2 else c1
        if rule == "up1":
            if c > c1:
                return j
        elif rule == "confirmed":
            v20 = _sma(volume, j, 20)
            vol_ok = (volume.iloc[j] > v20) if v20 == v20 else True
            if c > max(c1, c2) and vol_ok:
                return j
    return None


def _fwd(close, bench, pos, H):
    j = pos + H
    if j < len(close):
        f = close.iloc[j] / close.iloc[pos] - 1
        b = bench.iloc[j] / bench.iloc[pos] - 1
        return round(float(f), 4), round(float(f - b), 4)
    return None, None


def entry_timing_fields(close, volume, bench, ev):
    close = close.dropna().sort_index()
    volume = volume.reindex(close.index)
    bench = bench.reindex(close.index).ffill()
    i = ev["pos"]
    out = {}
    for rule in ("up1", "confirmed"):
        ep = find_entry(close, volume, i, rule)
        if ep is None:
            out[f"{rule}_conf"] = False
        else:
            f6, e6 = _fwd(close, bench, ep, H6)
            out[f"{rule}_conf"] = True
            out[f"{rule}_fwd6m"] = f6
            out[f"{rule}_exc6m"] = e6
            out[f"{rule}_days"] = ep - i
    return out


def entry_summary(df):
    """Isolate the entry-timing effect: for confirmed events, compare the
    confirmed-entry return to the same events' signal-close (baseline) return;
    and check whether the skipped (never-confirmed) events were duds."""
    out = {}
    for rule in ("up1", "confirmed"):
        cflag = f"{rule}_conf"
        if cflag not in df.columns:
            continue
        conf = df[(df[cflag] == True) & df[f"{rule}_fwd6m"].notna()]  # noqa: E712
        skipped = df[(df[cflag] == False) & df["fwd_6m"].notna()]  # noqa: E712
        out[rule] = {
            "n_confirmed": int(len(conf)),
            "n_skipped": int(len(skipped)),
            "median_days_to_entry": (round(float(df.loc[df[cflag] == True, f"{rule}_days"].median()), 1)
                                     if (df[cflag] == True).any() and f"{rule}_days" in df else None),  # noqa: E712
            "entry_6m": _stat(conf, f"{rule}_fwd6m", f"{rule}_exc6m"),
            "baseline_same_events_6m": _stat(conf, "fwd_6m", "excess_6m"),
            "skipped_events_baseline_6m": _stat(skipped, "fwd_6m", "excess_6m"),
        }
    return out


# ---- exit rules: fixed hold vs recover-to-pre-drop-price vs +trailing ------

EXIT_MAX = 126        # max holding window (trading days) for path simulation
TRAIL = 0.08          # trailing stop once the pre-drop target is hit


def exit_rules(close, ev):
    """Path-dependent exits, entering at the signal-day close (position i).
    Reference 'pre-drop price' = the close just BEFORE the drop window."""
    close = close.dropna().sort_index()
    i, w, n = ev["pos"], ev["window_len"], len(close)
    if i - w < 0 or i + 1 >= n:
        return {}
    entry = float(close.iloc[i])
    pre = float(close.iloc[i - w])                     # price before the overreaction
    end = min(i + EXIT_MAX, n - 1)
    path = close.iloc[i + 1:end + 1]
    if len(path) == 0 or entry <= 0:
        return {}
    out = {"pre_drop_target_ret": round(pre / entry - 1, 4),
           "mfe": round(float(path.max()) / entry - 1, 4)}
    # first day it recovers to the pre-drop price
    hit = None
    for k in range(len(path)):
        if float(path.iloc[k]) >= pre:
            hit = k + 1
            break
    out["recovered"] = hit is not None
    if hit is not None:
        out["days_to_recover"] = hit
        out["r_target"] = round(pre / entry - 1, 4)     # sell flat at target
        out["hold_target"] = hit
        # target + trailing stop: from the recovery day, ride until it drops TRAIL off its peak
        seg = close.iloc[i + hit:end + 1]
        peak = float(seg.iloc[0]); exitp = float(seg.iloc[-1]); hold = len(seg) - 1
        for k in range(1, len(seg)):
            c = float(seg.iloc[k]); peak = max(peak, c)
            if c < peak * (1 - TRAIL):
                exitp = c; hold = k; break
        out["r_trail"] = round(exitp / entry - 1, 4)
        out["hold_trail"] = hit + hold
    else:                                               # never recovered -> time-stop at window end
        out["r_target"] = round(float(path.iloc[-1]) / entry - 1, 4)
        out["hold_target"] = len(path)
        out["r_trail"] = out["r_target"]
        out["hold_trail"] = len(path)
    return out


def exit_summary(df):
    def ann(mean, hold):
        return round((1 + mean) ** (252.0 / hold) - 1, 4) if hold and mean > -1 else None
    def block(ret, hold_days_series_or_const, extra=None):
        r = ret.dropna()
        if len(r) == 0:
            return {"n": 0}
        hold = (float(hold_days_series_or_const.mean()) if hasattr(hold_days_series_or_const, "mean")
                else float(hold_days_series_or_const))
        d = {"n": int(len(r)), "mean": round(float(r.mean()), 4), "median": round(float(r.median()), 4),
             "hit": round(float((r > 0).mean()), 3), "avg_hold_days": round(hold, 1),
             "annualised": ann(float(r.mean()), hold)}
        if extra:
            d.update(extra)
        return d
    out = {"fixed_3m": block(df["fwd_3m"], 63), "fixed_6m": block(df["fwd_6m"], 126)}
    if "r_target" in df.columns:
        rec = round(float(df["recovered"].mean()), 3) if "recovered" in df else None
        out["target_predrop"] = block(df["r_target"], df["hold_target"], {"recover_rate": rec})
        out["target_plus_trail"] = block(df["r_trail"], df["hold_trail"], {"recover_rate": rec})
        out["median_days_to_recover"] = (round(float(df.loc[df["recovered"] == True, "days_to_recover"].median()), 1)  # noqa: E712
                                         if "days_to_recover" in df and (df.get("recovered") == True).any() else None)  # noqa: E712
    return out


# ---- fundamentals (indicative, yfinance) ---------------------------------

def ticker_fundamentals(yft):
    """Return a small helper structured per fiscal-year date, best-effort."""
    import yfinance as yf
    t = yf.Ticker(yft)
    try:
        inc = t.income_stmt              # columns = period end dates
        bal = t.balance_sheet
    except Exception:  # noqa: BLE001
        return None
    if inc is None or inc.empty:
        return None

    def row(df, *names):
        for nm in names:
            if nm in df.index:
                return df.loc[nm]
        return None
    ni = row(inc, "Net Income", "Net Income Common Stockholders")
    eq = row(bal, "Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity")
    debt = row(bal, "Total Debt")
    if debt is None:
        ld = row(bal, "Long Term Debt"); sd = row(bal, "Current Debt")
        debt = (ld.fillna(0) if ld is not None else 0) + (sd.fillna(0) if sd is not None else 0)
    # shares outstanding (single robust scalar) + sector, for earnings yield & industry
    shares_out, sector = None, None
    try:
        fi = t.fast_info
        shares_out = getattr(fi, "shares", None)
        if shares_out is None and hasattr(fi, "get"):
            shares_out = fi.get("shares")
    except Exception:  # noqa: BLE001
        pass
    try:
        info = t.get_info()
        if not shares_out:
            shares_out = info.get("sharesOutstanding")
        sector = info.get("sector") or info.get("sectorKey")
    except Exception:  # noqa: BLE001
        pass
    recs = []
    for d in ni.index:
        try:
            n_i = float(ni.get(d)); e_q = float(eq.get(d)) if eq is not None else np.nan
            dv = float(debt.get(d)) if hasattr(debt, "get") else float(debt)
            recs.append({"date": pd.Timestamp(d), "ni": n_i, "eq": e_q, "debt": dv})
        except Exception:  # noqa: BLE001
            continue
    return {"fy": sorted(recs, key=lambda r: r["date"]),
            "shares_out": float(shares_out) if shares_out else None, "sector": sector}


def metrics_for_event(fund, entry_date, entry_price, market):
    """Compute profitability/ROE/DE/earnings-yield-spread as-of the event."""
    if not fund or not fund["fy"]:
        return {}
    ed = pd.Timestamp(entry_date)
    prior = [r for r in fund["fy"] if r["date"] <= ed]
    fy = prior[-1] if prior else fund["fy"][0]
    ni, eq, debt = fy["ni"], fy["eq"], fy["debt"]
    m = {"profitable": bool(ni > 0) if ni == ni else None,
         "roe": round(ni / eq, 4) if (eq and eq == eq and eq != 0) else None,
         "de": round(debt / eq, 4) if (eq and eq == eq and eq != 0) else None}
    # earnings yield vs bond yield (uses a single shares-outstanding figure)
    shares_out = fund.get("shares_out")
    if shares_out and entry_price:
        ey = ni / (shares_out * entry_price)
        bond = AU_BOND_YIELD if market == "ASX" else metrics_for_event._us_bond.get(entry_date, 0.043)
        m["earnings_yield"] = round(ey, 4)
        m["ey_minus_bond"] = round(ey - bond, 4)
    return m
metrics_for_event._us_bond = {}


def load_us_bond():
    import yfinance as yf
    try:
        s = yf.download(US_BOND_TICKER, period="6y", interval="1d", progress=False)["Close"].dropna().squeeze()
        s = s / (10.0 if float(s.iloc[-1]) > 20 else 1.0) / 100.0   # -> decimal yield
        metrics_for_event._us_bond = {d.date().isoformat(): float(v) for d, v in s.items()}
        log(f"US 10y yield points: {len(metrics_for_event._us_bond)}")
    except Exception as e:  # noqa: BLE001
        log(f"US bond yield fetch failed: {e!r}")


# ---- summary / bucketing --------------------------------------------------

def summarise(df):
    out = {"n_events": int(len(df)), "by_metric": {}}
    for tag in ("3m", "6m"):
        col = f"fwd_{tag}"; ex = f"excess_{tag}"
        s = df[col].dropna()
        out[f"baseline_{tag}"] = _stat(df, col, ex)
    def buckets(name, series, edges, labels):
        res = {}
        cat = pd.cut(series, bins=edges, labels=labels)
        for lab in labels:
            sub = df[cat == lab]
            if len(sub) >= 5:
                res[str(lab)] = {"n": int(len(sub)),
                                 "3m": _stat(sub, "fwd_3m", "excess_3m"),
                                 "6m": _stat(sub, "fwd_6m", "excess_6m")}
        return res
    # profitability (boolean)
    out["by_metric"]["profitable"] = {}
    for lab, val in (("yes", True), ("no", False)):
        sub = df[df["profitable"] == val]
        if len(sub) >= 5:
            out["by_metric"]["profitable"][lab] = {"n": int(len(sub)),
                "3m": _stat(sub, "fwd_3m", "excess_3m"), "6m": _stat(sub, "fwd_6m", "excess_6m")}
    out["by_metric"]["roe"] = buckets("roe", df["roe"], [-np.inf, 0, .05, .10, .15, np.inf],
                                      ["<0", "0-5%", "5-10%", "10-15%", ">15%"])
    out["by_metric"]["de"] = buckets("de", df["de"], [-np.inf, .3, .75, 1.25, np.inf],
                                     ["<0.3", "0.3-0.75", "0.75-1.25", ">1.25"])
    if "ey_minus_bond" in df:
        out["by_metric"]["ey_minus_bond"] = buckets("eyb", df["ey_minus_bond"],
            [-np.inf, 0, .03, .06, np.inf], ["<0", "0-3%", "3-6%", ">6%"])
    out["by_metric"]["momentum_6m"] = buckets("mom", df["momentum_6m"],
        [-np.inf, -.1, 0, .2, np.inf], ["<-10%", "-10-0%", "0-20%", ">20%"])
    # drop-size buckets — the strongest lever
    out["by_metric"]["drop_size"] = buckets("raw", df["raw"],
        [-np.inf, -.30, -.20, -.15, -.10], [">30%", "20-30%", "15-20%", "10-15%"])
    # industry / sector breakdown (>=20 events per sector)
    if "sector" in df.columns and df["sector"].notna().any():
        out["by_sector"] = {}
        for sec, sub in df.groupby("sector"):
            if len(sub) >= 20:
                out["by_sector"][str(sec)] = {"n": int(len(sub)),
                    "3m": _stat(sub, "fwd_3m", "excess_3m"), "6m": _stat(sub, "fwd_6m", "excess_6m")}
    return out


def _stat(df, col, ex):
    s = df[col].dropna()
    if len(s) == 0:
        return {"n": 0}
    e = df[ex].dropna()
    return {"n": int(len(s)), "mean": round(float(s.mean()), 4), "median": round(float(s.median()), 4),
            "hit_rate": round(float((s > 0).mean()), 3), "std": round(float(s.std()), 4),
            "mean_excess": round(float(e.mean()), 4) if len(e) else None,
            "ir_like": round(float(s.mean() / s.std()), 3) if s.std() else None}


# ---- orchestration --------------------------------------------------------

def run(markets, years, data_dir, limit=0):
    load_us_bond()
    rows = []
    period = f"{int(years*365)+260}d"
    for name in markets:
        mcfg = cfg.market_params(name)
        fb = os.path.join(data_dir, f"universe_{name}.csv")
        uni = datamod.get_universe(mcfg, local_fallback=fb if os.path.exists(fb) else None)
        if limit:
            uni = uni.head(limit)
        prices = datamod.download_prices(list(uni["yahoo"]), period_days=int(years*365)+260)
        benches = datamod.download_benchmarks(mcfg, period_days=int(years*365)+260)
        bench = benches.get(mcfg.BENCHMARK_300)      # None or a pandas Series
        if bench is None:
            bench = benches.get(mcfg.BENCHMARK_200)
        if bench is None:
            log(f"{name}: no benchmark, skipping"); continue
        name_by = dict(zip(uni["yahoo"], uni["name"])); code_by = dict(zip(uni["yahoo"], uni["code"]))
        sector_by = dict(zip(uni["yahoo"], uni["sector"])) if "sector" in uni.columns else {}
        # Stage 1: events per ticker (cheap, price only)
        evmap = {}
        for y, df in prices.items():
            evs = detect_events(df["Close"], df["Volume"], bench, mcfg)
            if evs:
                evmap[y] = evs
        log(f"{name}: {sum(len(v) for v in evmap.values())} events across {len(evmap)} names")
        # Stage 2: fundamentals only for names with events
        for y, evs in evmap.items():
            fund = None
            try:
                fund = ticker_fundamentals(y)
            except Exception:  # noqa: BLE001
                fund = None
            df = prices[y]
            sec = sector_by.get(y)
            for ev in evs:
                fm = forward_and_momentum(df["Close"], bench, ev)
                et = entry_timing_fields(df["Close"], df["Volume"], bench, ev)
                xr = exit_rules(df["Close"], ev)
                mm = metrics_for_event(fund, ev["date"], ev["entry"], name) if fund else {}
                rows.append({"market": name, "ticker": code_by.get(y, y), "name": name_by.get(y, y),
                             "sector": sec or (fund.get("sector") if fund else None),
                             "date": ev["date"], "window_len": ev["window_len"], "raw": ev["raw"],
                             "z": ev["z"], **fm, **et, **xr, **mm})
    df = pd.DataFrame(rows)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "backtest_events.csv"), index=False)
    summary = summarise(df) if len(df) else {"n_events": 0}
    if len(df):
        summary["entry_timing"] = entry_summary(df)
        summary["exit_rules"] = exit_summary(df)
    summary["caveats"] = ["Universe = today's membership; delisted excluded (survivorship bias).",
                          "Fundamentals: yfinance, survivor-only & as-restated, ~4y deep — INDICATIVE, not point-in-time.",
                          "AU 10y bond yield is a fixed proxy (%.1f%%)." % (AU_BOND_YIELD*100)]
    with open(os.path.join(data_dir, "backtest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {len(df)} events; summary saved")
    print(json.dumps({k: summary[k] for k in ("n_events", "baseline_3m", "baseline_6m") if k in summary}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--markets", default="")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        from . import test_factor  # noqa: F401
        return
    markets = [m.strip() for m in a.markets.split(",") if m.strip()] or list(cfg.MARKETS)
    run(markets, a.years, a.data_dir, a.limit)


if __name__ == "__main__":
    main()
