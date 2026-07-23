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

import argparse, json, math, os, re, sys
import numpy as np
import pandas as pd

from . import config as cfg
from . import data as datamod
from . import us_insiders
from . import eodhd

H3, H6 = 63, 126           # ~3 and ~6 trading months
# Fixed-hold horizons (trading days) for the hold-period study: ~3, 4, 6, 8 weeks.
# CLEAN unconditional holds (unlike the ce_te* recovery-or-cut exits) so we can test
# whether a shorter hold recycles the credit facility faster for a higher return on it.
HOLD_GRID_D = (15, 21, 30, 42)
COOLDOWN = 21              # trading days between distinct events for one name
# Backtest keeps a WIDER 10% net than the live screen's 15% floor, so we can still
# bucket by drop size and re-confirm the deeper-is-better finding on future runs.
BT_DROP_FLOOR = -0.10
# Emission z floor — LOOSER than the analysis gate (cfg.Z_THRESHOLD = -2.5) so the
# milder-dislocation events land in the file and z can be TOGGLED post-hoc without a
# re-run. The approved analysis still applies z <= -2.5 (see GATES.md); this only
# controls what gets WRITTEN. The live screen is unaffected — it uses cfg.Z_THRESHOLD.
BT_EMIT_Z = -1.5
MOM_LOOKBACK = 126        # 6m pre-drop momentum
US_BOND_TICKER = "^TNX"    # US 10y yield (yahoo)
AU_BOND_YIELD = 0.043     # AU 10y proxy (flagged; no clean free daily series)


def log(m): print(f"[factor] {m}", file=sys.stderr, flush=True)


# ---- vectorised event detection ------------------------------------------

def to_naive_daily(s):
    """Force a price/benchmark Series onto a tz-naive, midnight-normalised daily
    DatetimeIndex. Critical when mixing sources: EODHD prices are tz-naive dates
    but yfinance daily bars can be tz-AWARE (US/Eastern). A reindex across that
    mismatch silently yields all-NaN, so every index-relative drop is NaN and NO
    event can ever fire (a hard zero). Normalising both sides first prevents it."""
    if s is None:
        return None
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    out = pd.Series(np.asarray(s.values), index=idx)
    return out[~out.index.duplicated(keep="last")].sort_index()


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
        cond = ((z <= BT_EMIT_Z) & (wret <= BT_DROP_FLOOR) & (wret >= cfg.MAX_DROP_FLOOR) &
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
        _adv = adv.iloc[i]
        events.append({"pos": i, "date": close.index[i].date().isoformat(),
                       "window_len": w, "z": round(float(best_z.iloc[i]), 3),
                       "raw": round(float(close.iloc[i] / close.iloc[i - w] - 1), 4),
                       "entry": float(close.iloc[i]),
                       "avg_daily_value": (round(float(_adv), 0) if pd.notna(_adv) else None)})
    return events


def forward_and_momentum(close, bench, ev):
    """Attach 3m/6m forward + excess returns and pre-drop momentum to an event."""
    close = close.dropna().sort_index()
    bench = bench.reindex(close.index).ffill()
    i, w, n = ev["pos"], ev["window_len"], len(close)
    out = {}
    entry_px = float(close.iloc[i])
    for tag, H in (("3m", H3), ("6m", H6)):
        j = i + H
        b0 = float(bench.iloc[i]) if j < n else 0.0
        if j < n and entry_px > 0 and b0 > 0:      # guard: near-zero entry -> Inf return
            fwd = close.iloc[j] / entry_px - 1
            bfwd = bench.iloc[j] / b0 - 1
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
            f3, e3 = _fwd(close, bench, ep, H3)
            out[f"{rule}_conf"] = True
            out[f"{rule}_fwd6m"] = f6
            out[f"{rule}_exc6m"] = e6
            out[f"{rule}_fwd3m"] = f3          # confirmed-entry 3-month hold (the exact stack)
            out[f"{rule}_exc3m"] = e3
            out[f"{rule}_days"] = ep - i
            if rule == "confirmed":            # clean fixed-hold returns for the hold-period study
                for N in HOLD_GRID_D:
                    fN, eN = _fwd(close, bench, ep, N)
                    out[f"confirmed_fwd{N}d"] = fN
                    out[f"confirmed_exc{N}d"] = eN
    return out


def entry_summary(df):
    """Isolate the entry-timing effect: for confirmed events, compare the
    confirmed-entry return to the same events' signal-close (baseline) return;
    and check whether the skipped (never-confirmed) events were duds."""
    out = {}
    for rule in ("up1", "confirmed"):
        cflag = f"{rule}_conf"
        if cflag not in df.columns or f"{rule}_fwd6m" not in df.columns:
            # No event ever confirmed this entry rule (column never created).
            continue
        conf = df[(df[cflag] == True) & df[f"{rule}_fwd6m"].notna()]  # noqa: E712
        skipped = df[(df[cflag] == False) & df["fwd_6m"].notna()]  # noqa: E712
        out[rule] = {
            "n_confirmed": int(len(conf)),
            "n_skipped": int(len(skipped)),
            "median_days_to_entry": (round(float(df.loc[df[cflag] == True, f"{rule}_days"].median()), 1)
                                     if (df[cflag] == True).any() and f"{rule}_days" in df else None),  # noqa: E712
            "entry_6m": _stat(conf, f"{rule}_fwd6m", f"{rule}_exc6m"),
            "entry_3m": _stat(conf, f"{rule}_fwd3m", f"{rule}_exc3m") if f"{rule}_fwd3m" in df else {"n": 0},
            "baseline_same_events_6m": _stat(conf, "fwd_6m", "excess_6m"),
            "baseline_same_events_3m": _stat(conf, "fwd_3m", "excess_3m"),
            "skipped_events_baseline_6m": _stat(skipped, "fwd_6m", "excess_6m"),
            "skipped_events_baseline_3m": _stat(skipped, "fwd_3m", "excess_3m"),
        }
    return out


# ---- exit rules: fixed hold vs recover-to-pre-drop-price vs +trailing ------

EXIT_MAX = 126        # max window for a winner to ride via the trailing stop
RECOVER_TIMESTOP = 63  # cut a name that hasn't recovered to pre-drop by ~3 months
TRAIL = 0.08          # trailing stop once the pre-drop target is hit
STOPS = (5, 8, 10, 12, 25, 30)  # hard stop-loss levels (% below entry), fixed-3m base.
                                # 25/30 = WIDE 'catastrophe' stops: they should not fire on
                                # normal mean-reversion noise (tight 5-12% ones do, and hurt),
                                # but cap the true blow-up / delisting tail — the survivorship
                                # insurance without killing the return.

TIME_EXIT_GRID = (15, 21, 30, 42)  # candidate non-recovery time-stops (trading days) to sweep


def confirmed_exits(close, bench, entry_pos, pre, grid=TIME_EXIT_GRID):
    """Exit variants anchored at the CONFIRMED entry (not the signal close), so the
    numbers match the model we actually trade. Returns, per event:
      ce_trail        — ride a trailing stop once the price recovers to its pre-drop
                        level (time-stop at 3m if it never recovers). No early cut.
      ce_te{N}        — the recovery TIME-EXIT: if the name reclaims its pre-drop price
                        within N days of entry, ride the trailing stop; else cut it at
                        day N. This is the 'don't wait forever for a bounce' rule.
    Each has a matching _exc (excess vs benchmark over the realised hold)."""
    close = close.dropna().sort_index()
    bench = bench.reindex(close.index).ffill()      # align to stock's dates (positions match close)
    n = len(close); p = int(entry_pos)
    if p < 0 or p + 1 >= n:
        return {}
    e = float(close.iloc[p])
    if e <= 0:
        return {}
    end = min(p + EXIT_MAX, n - 1)
    b0 = float(bench.iloc[p]) if (p < len(bench) and pd.notna(bench.iloc[p])) else 0.0

    def bexc(exit_pos, r):
        if exit_pos is None or exit_pos >= n or b0 <= 0:
            return None
        b = float(bench.iloc[exit_pos]) / b0 - 1
        return round(r - b, 4)

    def trail_from(rec_day):
        seg = close.iloc[rec_day:end + 1]
        peak = float(seg.iloc[0]); exitp = float(seg.iloc[-1]); hold = len(seg) - 1
        for k in range(1, len(seg)):
            c = float(seg.iloc[k]); peak = max(peak, c)
            if c < peak * (1 - TRAIL):
                exitp = c; hold = k; break
        return exitp / e - 1, rec_day + hold

    # first day the pre-drop price is reclaimed, within the 3-month recovery window
    rec_end = min(p + RECOVER_TIMESTOP, n - 1)
    rec = None
    for j in range(p + 1, rec_end + 1):
        if float(close.iloc[j]) >= pre:
            rec = j; break

    out = {}
    if rec is not None:
        r, xp = trail_from(rec)
    else:
        xp = rec_end; r = float(close.iloc[xp]) / e - 1
    out["ce_trail"] = round(r, 4); out["ce_trail_exc"] = bexc(xp, r)

    for N in grid:
        npos = min(p + N, n - 1)
        if rec is not None and (rec - p) <= N:
            r, xp = trail_from(rec)                 # bounced in time -> ride it
        else:
            xp = npos; r = float(close.iloc[xp]) / e - 1   # no bounce by day N -> cut
        out[f"ce_te{N}"] = round(r, 4); out[f"ce_te{N}_exc"] = bexc(xp, r)

    out["ce_rec_days"] = (rec - p) if rec is not None else -1
    return out


def exit_rules(close, ev):
    """Path-dependent exits, entering at the signal-day close (position i).
    Reference 'pre-drop price' = the close just BEFORE the drop window.
    Non-recoverers are cut at a 3-month time-stop; winners ride a trailing stop.
    Also simulates a fixed-3-month hold with a hard stop-loss at several levels."""
    close = close.dropna().sort_index()
    i, w, n = ev["pos"], ev["window_len"], len(close)
    if i - w < 0 or i + 1 >= n:
        return {}
    entry = float(close.iloc[i])
    pre = float(close.iloc[i - w])
    if entry <= 0:
        return {}
    end = min(i + EXIT_MAX, n - 1)
    out = {"pre_drop_target_ret": round(pre / entry - 1, 4),
           "mfe": round(float(close.iloc[i + 1:end + 1].max()) / entry - 1, 4)}

    # recover to pre-drop within the 3-month time-stop window
    rec_end = min(i + RECOVER_TIMESTOP, n - 1)
    hit = None
    for j in range(i + 1, rec_end + 1):
        if float(close.iloc[j]) >= pre:
            hit = j
            break
    out["recovered"] = hit is not None
    if hit is not None:
        out["days_to_recover"] = hit - i
        out["r_target"] = round(pre / entry - 1, 4)
        out["hold_target"] = hit - i
        seg = close.iloc[hit:end + 1]                  # ride the winner from recovery day
        peak = float(seg.iloc[0]); exitp = float(seg.iloc[-1]); hold = len(seg) - 1
        for k in range(1, len(seg)):
            c = float(seg.iloc[k]); peak = max(peak, c)
            if c < peak * (1 - TRAIL):
                exitp = c; hold = k; break
        out["r_trail"] = round(exitp / entry - 1, 4)
        out["hold_trail"] = (hit - i) + hold
    else:                                              # 3-month time-stop
        tp = min(i + RECOVER_TIMESTOP, n - 1)
        out["r_target"] = round(float(close.iloc[tp]) / entry - 1, 4)
        out["hold_target"] = tp - i
        out["r_trail"] = out["r_target"]; out["hold_trail"] = tp - i

    # fixed-3-month hold WITH a hard stop-loss at each level
    h3 = min(i + 63, n - 1)
    for stop in STOPS:
        lvl = entry * (1 - stop / 100.0)
        ex, exd = None, None
        for j in range(i + 1, h3 + 1):
            if float(close.iloc[j]) <= lvl:
                ex, exd = float(close.iloc[j]), j - i
                break
        if ex is None:
            ex, exd = float(close.iloc[h3]), h3 - i
        out[f"r_sl{stop}"] = round(ex / entry - 1, 4)
        out[f"hold_sl{stop}"] = exd
    return out


def exit_summary(df):
    def ann(mean, hold):
        return round((1 + mean) ** (252.0 / hold) - 1, 4) if hold and mean > -1 else None

    def block(ret, hold, extra=None):
        r = ret.dropna()
        if len(r) == 0:
            return {"n": 0}
        h = float(hold.mean()) if hasattr(hold, "mean") else float(hold)
        d = {"n": int(len(r)), "mean": round(float(r.mean()), 4), "median": round(float(r.median()), 4),
             "hit": round(float((r > 0).mean()), 3), "avg_hold_days": round(h, 1),
             "annualised": ann(float(r.mean()), h)}
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
    for stop in STOPS:
        if f"r_sl{stop}" in df.columns:
            out[f"fixed3m_stop{stop}"] = block(df[f"r_sl{stop}"], df[f"hold_sl{stop}"])
    return out


def _fav(s):
    return cfg.is_favoured(s)      # single source of truth (config), consistent across markets


def _avoid(s):
    return cfg.is_avoided(s)


def _quality_mask(df):
    """Row mask for the profitability gate. When cfg.REQUIRE_QUALITY, keep only events
    whose NPAT and FCF margins clear the floors; missing fundamentals (NaN) fail closed.
    When the columns are absent (e.g. yfinance runs with no fundamentals), pass all."""
    if not cfg.REQUIRE_QUALITY or "npat_margin" not in df.columns or "fcf_margin" not in df.columns:
        return pd.Series(True, index=df.index)
    npat = pd.to_numeric(df["npat_margin"], errors="coerce")
    fcf = pd.to_numeric(df["fcf_margin"], errors="coerce")
    return (npat >= cfg.NPAT_MARGIN_MIN) & (fcf >= cfg.FCF_MARGIN_MIN)


def _trigger_mask(df):
    """Exclude structural-repricing triggers when the events are tagged (trigger_primary
    present). Fails OPEN: 'none'/untagged rows pass — only positively-identified
    structural drops are dropped. When the column is absent, pass all."""
    if not cfg.EXCLUDE_STRUCTURAL_TRIGGERS or "trigger_primary" not in df.columns:
        return pd.Series(True, index=df.index)
    return ~df["trigger_primary"].isin(cfg.STRUCTURAL_TRIGGERS)


# ---- insider (SEC Form 4) signal summary — US only ------------------------

def insider_summary(df):
    """Do prior director/insider trades sort forward returns among oversold names?
    Buckets by the coarse buy/sell/none signal and by the director-buy flag (the
    live 'double the position' trigger), and estimates the effect of doubling the
    weight on director-buy events versus equal-weighting everything."""
    if "insider_signal" not in df.columns:
        return {"note": "no insider data in this run"}
    d = df[df["insider_signal"].notna()]
    if len(d) == 0:
        return {"note": "no insider signal captured"}
    out = {"n_with_signal": int(len(d))}
    out["by_signal"] = {}
    for lab in ("buy", "sell", "none", "mixed"):
        sub = d[d["insider_signal"] == lab]
        if len(sub) >= 5:
            out["by_signal"][lab] = {"n": int(len(sub)),
                "3m": _stat(sub, "fwd_3m", "excess_3m"), "6m": _stat(sub, "fwd_6m", "excess_6m")}
    for flag_col in ("director_buy", "director_sell"):
        if flag_col in d.columns:
            blk = {}
            for lab, val in (("yes", True), ("no", False)):
                sub = d[d[flag_col] == val]
                if len(sub) >= 5:
                    blk[lab] = {"n": int(len(sub)),
                        "3m": _stat(sub, "fwd_3m", "excess_3m"), "6m": _stat(sub, "fwd_6m", "excess_6m")}
            out[flag_col] = blk
    # doubling-up: 2x weight on director-buy events vs equal weight
    for tag in ("3m", "6m"):
        col = f"fwd_{tag}"
        sub = d[d[col].notna()]
        if len(sub) >= 10 and "director_buy" in sub.columns:
            w = sub["director_buy"].map(lambda b: 2.0 if b else 1.0)
            out[f"double_on_director_buy_{tag}"] = {
                "equal_weight": round(float(sub[col].mean()), 4),
                "double_dirbuy": round(float((sub[col] * w).sum() / w.sum()), 4),
                "n_dirbuy": int((sub["director_buy"] == True).sum())}  # noqa: E712
    return out


def by_horizon_summary(df):
    """Trailing 5 / 10 / 15-year annualised return of the favoured+floor cohort vs
    the market — to test whether the edge is consistent across regimes rather than
    a one-off crisis effect. ONLY meaningful on a survivorship-free universe."""
    if "year" not in df.columns or "sector" not in df.columns:
        return {}
    fav = df[df["sector"].map(_fav) & (df["raw"] <= cfg.ABS_DROP_THRESHOLD)
             & df["fwd_3m"].notna() & _quality_mask(df) & _trigger_mask(df)]
    if len(fav) == 0:
        return {}
    ymax = int(fav["year"].max())
    ann = lambda r: (1 + r) ** 4 - 1  # noqa: E731
    out = {"quality_gate": bool(cfg.REQUIRE_QUALITY)}
    for h in (5, 10, 15):
        sub = fav[fav["year"] >= ymax - h + 1]
        if len(sub) >= 20:
            m = float(_winsorise(sub["fwd_3m"]).mean())
            idx = float(_winsorise(sub["fwd_3m"] - sub["excess_3m"]).mean())
            out[f"{h}y"] = {"n": int(len(sub)), "window": f"{ymax-h+1}-{ymax}",
                            "strat_ann": round(ann(m), 4), "mkt_ann": round(ann(idx), 4),
                            "excess_ann": round(ann(m) - ann(idx), 4),
                            "hit": round(float((sub["fwd_3m"] > 0).mean()), 3)}
    return out


def by_year_summary(df):
    """Forward-return distribution by the event's calendar year (needs >=10/yr).
    Lets us report per-year and 5-year-average returns once the run completes."""
    if "year" not in df.columns:
        return {}
    out = {}
    for yr, sub in df.groupby("year"):
        if len(sub) >= 10:
            out[str(int(yr))] = {"n": int(len(sub)),
                "3m": _stat(sub, "fwd_3m", "excess_3m"), "6m": _stat(sub, "fwd_6m", "excess_6m")}
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
    try:
        cf = t.cashflow
    except Exception:  # noqa: BLE001
        cf = None

    def row(df, *names):
        if df is None:
            return None
        for nm in names:
            if nm in df.index:
                return df.loc[nm]
        return None
    ni = row(inc, "Net Income", "Net Income Common Stockholders")
    eq = row(bal, "Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity")
    rev = row(inc, "Total Revenue", "Operating Revenue")
    eps = row(inc, "Diluted EPS", "Basic EPS")
    fcf = row(cf, "Free Cash Flow")
    if fcf is None:                      # fall back to operating CF - capex
        ocf = row(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
        capex = row(cf, "Capital Expenditure")
        if ocf is not None:
            fcf = ocf + (capex.fillna(0) if capex is not None else 0)  # capex stored negative
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
    try:
        divs = t.dividends              # tz-aware Series indexed by pay date
    except Exception:  # noqa: BLE001
        divs = None

    def _g(series, d):
        try:
            return float(series.get(d)) if series is not None else np.nan
        except Exception:  # noqa: BLE001
            return np.nan
    recs = []
    for d in ni.index:
        try:
            recs.append({"date": pd.Timestamp(d), "ni": float(ni.get(d)),
                         "eq": (float(eq.get(d)) if eq is not None else np.nan),
                         "debt": (float(debt.get(d)) if hasattr(debt, "get") else float(debt)),
                         "rev": _g(rev, d), "eps": _g(eps, d), "fcf": _g(fcf, d)})
        except Exception:  # noqa: BLE001
            continue
    return {"fy": sorted(recs, key=lambda r: r["date"]),
            "shares_out": float(shares_out) if shares_out else None,
            "sector": sector, "dividends": divs}


def _as_iso(x):
    if x is None:
        return None
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:  # noqa: BLE001
        s = str(x)[:10]
        return s if len(s) == 10 else None


def _pos(x):
    return x is not None and x == x and x != 0


def metrics_for_event(fund, entry_date, entry_price, market):
    """Profitability / ROE / DE / earnings-yield + the five quality factors, taken
    POINT-IN-TIME: the latest fiscal year whose FILING date is on/before the event
    (EODHD supplies filing_date; yfinance falls back to the fiscal-period date)."""
    if not fund or not fund.get("fy"):
        return {}
    ed_iso = _as_iso(entry_date)

    def known(r):                      # date the statement became public
        return _as_iso(r.get("filing_date") or r.get("date"))
    prior = [r for r in fund["fy"] if known(r) and ed_iso and known(r) <= ed_iso]
    fy = prior[-1] if prior else fund["fy"][0]
    ni, eq, debt = fy.get("ni"), fy.get("eq"), fy.get("debt")
    m = {"profitable": (bool(ni > 0) if ni is not None else None),
         "roe": round(ni / eq, 4) if (ni is not None and _pos(eq)) else None,
         "de": round(debt / eq, 4) if (debt is not None and _pos(eq)) else None}
    shares_out = fund.get("shares_out")
    if shares_out and entry_price:
        m["market_cap"] = round(float(shares_out) * float(entry_price), 0)  # cap at the event
    if shares_out and entry_price and ni is not None:
        ey = ni / (shares_out * entry_price)
        bond = AU_BOND_YIELD if market == "ASX" else metrics_for_event._us_bond.get(entry_date, 0.043)
        m["earnings_yield"] = round(ey, 4)
        m["ey_minus_bond"] = round(ey - bond, 4)

    # ---- the five quality factors under test (point-in-time) ----
    rev, fcf = fy.get("rev"), fy.get("fcf")
    if _pos(rev):
        if ni is not None:
            m["npat_margin"] = round(ni / rev, 4)                   # profit margin
        if fcf is not None:
            m["fcf_margin"] = round(fcf / rev, 4)                   # free-cash-flow margin
    # growth over the last ~5 filed fiscal years (6 points = 5-year CAGR)
    m["rev_growth"] = _cagr([r.get("rev") for r in prior[-6:]])
    m["eps_growth"] = _cagr([r.get("eps") for r in prior[-6:]])
    dy = _ttm_div_yield(fund.get("dividends"), ed_iso, entry_price)
    if dy is not None:
        m["div_yield"] = dy
    return m
metrics_for_event._us_bond = {}


def _ttm_div_yield(divs, ed_iso, price):
    """Trailing-12-month dividend / price. Accepts EODHD's [(iso_date, value)] or a
    yfinance pandas Series. None if no dividend history at all."""
    if divs is None or not price or ed_iso is None or (hasattr(divs, "__len__") and len(divs) == 0):
        return None
    lo_iso = (pd.Timestamp(ed_iso) - pd.Timedelta(days=365)).date().isoformat()
    try:
        if isinstance(divs, list):                                 # [(iso_date, value)] (EODHD)
            ttm = sum(v for d, v in divs if lo_iso < str(d)[:10] <= ed_iso)
        else:                                                      # pandas Series (yfinance)
            idx = divs.index
            tz = getattr(idx, "tz", None)
            lo = pd.Timestamp(lo_iso); hi = pd.Timestamp(ed_iso)
            if tz is not None:
                lo, hi = lo.tz_localize(tz), hi.tz_localize(tz)
            ttm = float(divs[(idx > lo) & (idx <= hi)].sum())
        return round(ttm / price, 4)
    except Exception:  # noqa: BLE001
        return None


def _cagr(vals):
    """CAGR from first to last POSITIVE value in an ordered fiscal-year list, using
    the actual year-gap between them (so a loss year in the middle doesn't distort
    the span). Needs >=2 positive points >=1 year apart; else None."""
    idx = [i for i, v in enumerate(vals) if v is not None and v == v and v > 0]
    if len(idx) < 2:
        return None
    i0, i1 = idx[0], idx[-1]
    yrs = i1 - i0
    if yrs < 1:
        return None
    try:
        return round((vals[i1] / vals[i0]) ** (1.0 / yrs) - 1.0, 4)
    except Exception:  # noqa: BLE001
        return None


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
        if col not in df.columns or ex not in df.columns:
            continue
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
    # profitability (boolean). Guarded: a quota-starved / fundamentals-less run may
    # have no 'profitable' column at all — skip rather than KeyError-crash.
    if "profitable" in df.columns:
        out["by_metric"]["profitable"] = {}
        for lab, val in (("yes", True), ("no", False)):
            sub = df[df["profitable"] == val]
            if len(sub) >= 5:
                out["by_metric"]["profitable"][lab] = {"n": int(len(sub)),
                    "3m": _stat(sub, "fwd_3m", "excess_3m"), "6m": _stat(sub, "fwd_6m", "excess_6m")}
    if "roe" in df.columns:
        out["by_metric"]["roe"] = buckets("roe", df["roe"], [-np.inf, 0, .05, .10, .15, np.inf],
                                          ["<0", "0-5%", "5-10%", "10-15%", ">15%"])
    if "de" in df.columns:
        out["by_metric"]["de"] = buckets("de", df["de"], [-np.inf, .3, .75, 1.25, np.inf],
                                         ["<0.3", "0.3-0.75", "0.75-1.25", ">1.25"])
    if "ey_minus_bond" in df:
        out["by_metric"]["ey_minus_bond"] = buckets("eyb", df["ey_minus_bond"],
            [-np.inf, 0, .03, .06, np.inf], ["<0", "0-3%", "3-6%", ">6%"])
    if "momentum_6m" in df.columns:
        out["by_metric"]["momentum_6m"] = buckets("mom", df["momentum_6m"],
            [-np.inf, -.1, 0, .2, np.inf], ["<-10%", "-10-0%", "0-20%", ">20%"])
    # the five quality factors under test — each bucket reports hit-rate AND mean/excess
    for fac, edges, labs in (
        ("npat_margin", [-np.inf, 0, .05, .10, .20, np.inf], ["<0", "0-5%", "5-10%", "10-20%", ">20%"]),
        ("fcf_margin",  [-np.inf, 0, .05, .15, np.inf],      ["<0", "0-5%", "5-15%", ">15%"]),
        ("rev_growth",  [-np.inf, 0, .10, .25, np.inf],      ["<0", "0-10%", "10-25%", ">25%"]),
        ("eps_growth",  [-np.inf, 0, .10, .25, np.inf],      ["<0", "0-10%", "10-25%", ">25%"]),
        ("div_yield",   [-np.inf, .0001, .02, .04, np.inf],  ["none", "0-2%", "2-4%", ">4%"]),
    ):
        if fac in df.columns and df[fac].notna().any():
            out["by_metric"][fac] = buckets(fac, df[fac], edges, labs)
    # drop-size buckets — the strongest lever
    if "raw" in df.columns:
        out["by_metric"]["drop_size"] = buckets("raw", df["raw"],
            [-np.inf, -.30, -.20, -.15, -.10], [">30%", "20-30%", "15-20%", "10-15%"])
    # industry / sector breakdown (>=20 events per sector)
    if "sector" in df.columns and df["sector"].notna().any():
        out["by_sector"] = {}
        for sec, sub in df.groupby("sector"):
            if len(sub) >= 20:
                out["by_sector"][str(sec)] = {"n": int(len(sub)),
                    "3m": _stat(sub, "fwd_3m", "excess_3m"), "6m": _stat(sub, "fwd_6m", "excess_6m")}
    # exchange breakdown — does the added Nasdaq cohort behave like the S&P one?
    if "exchange" in df.columns and df["exchange"].notna().any():
        out["by_exchange"] = {}
        fav_mask = df["sector"].map(_fav) if "sector" in df.columns else None
        for ex, sub in df.groupby("exchange"):
            if len(sub) >= 20:
                blk = {"n": int(len(sub)), "3m": _stat(sub, "fwd_3m", "excess_3m"),
                       "6m": _stat(sub, "fwd_6m", "excess_6m")}
                if fav_mask is not None:
                    favsub = sub[sub["sector"].map(_fav)]
                    if len(favsub) >= 10:
                        blk["favoured_3m"] = _stat(favsub, "fwd_3m", "excess_3m")
                out["by_exchange"][str(ex)] = blk
    return out


def _winsorise(s, lo=1, hi=99):
    """Clip to [1,99] pct so a few relist/discontinuity artifacts can't dominate a
    mean. Robust to a survivorship-free universe where dead names produce fat tails."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 20:
        return s
    a, b = np.nanpercentile(s, [lo, hi])
    return s.clip(a, b)


def _stat(df, col, ex):
    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"n": 0}
    e = df[ex].replace([np.inf, -np.inf], np.nan).dropna()
    sw = _winsorise(s)                     # winsorised mean/std; hit-rate & median stay raw
    ew = _winsorise(e)
    return {"n": int(len(s)), "mean": round(float(sw.mean()), 4), "median": round(float(s.median()), 4),
            "hit_rate": round(float((s > 0).mean()), 3), "std": round(float(sw.std()), 4),
            "mean_excess": round(float(ew.mean()), 4) if len(ew) else None,
            "ir_like": round(float(sw.mean() / sw.std()), 3) if sw.std() else None}


# ---- orchestration --------------------------------------------------------

def run(markets, years, data_dir, limit=0):
    load_us_bond()
    rows = []
    period = f"{int(years*365)+260}d"
    for name in markets:
        mcfg = cfg.market_params(name)
        # Nasdaq widening was tested and shelved (neutral Sharpe, +40% workload) — backtest
        # uses S&P 1500 like live. To re-test Nasdaq, set mcfg.UNIVERSE = "us_expanded" here.
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
        exch_by = dict(zip(uni["yahoo"], uni["exchange"])) if "exchange" in uni.columns else {}
        # Stage 1: events per ticker (cheap, price only)
        evmap = {}
        for y, df in prices.items():
            evs = detect_events(df["Close"], df["Volume"], bench, mcfg)
            if evs:
                evmap[y] = evs
        log(f"{name}: {sum(len(v) for v in evmap.values())} events across {len(evmap)} names")
        # Insider (SEC Form 4) signal — US only, fetched once per event-ticker.
        insider_map = {}
        if name == "US" and evmap:
            ev_by_tk = {}
            for y, evs in evmap.items():
                tk = code_by.get(y, y)
                ev_by_tk.setdefault(tk, []).extend(e["date"] for e in evs)
            try:
                insider_map = us_insiders.insider_signals_for_events(ev_by_tk, years=years)
            except Exception as e:  # noqa: BLE001
                log(f"insider signal fetch failed: {e!r}")
        # Stage 2: fundamentals only for names with events
        use_eodhd = bool(eodhd.token())
        if use_eodhd:
            log(f"{name}: fundamentals via EODHD (point-in-time)")
        for y, evs in evmap.items():
            fund = None
            try:
                fund = (eodhd.fundamentals(code_by.get(y, y), name) if use_eodhd
                        else ticker_fundamentals(y))
            except Exception:  # noqa: BLE001
                fund = None
            df = prices[y]
            sec = sector_by.get(y)
            for ev in evs:
                fm = forward_and_momentum(df["Close"], bench, ev)
                et = entry_timing_fields(df["Close"], df["Volume"], bench, ev)
                xr = exit_rules(df["Close"], ev)
                mm = metrics_for_event(fund, ev["date"], ev["entry"], name) if fund else {}
                ins = insider_map.get((code_by.get(y, y), ev["date"]), {}) if name == "US" else {}
                rows.append({"market": name, "ticker": code_by.get(y, y), "name": name_by.get(y, y),
                             "sector": sec or (fund.get("sector") if fund else None),
                             "exchange": exch_by.get(y),
                             "date": ev["date"], "year": int(ev["date"][:4]),
                             "window_len": ev["window_len"], "raw": ev["raw"],
                             "z": ev["z"], "avg_daily_value": ev.get("avg_daily_value"),
                             **fm, **et, **xr, **mm, **ins})
    _finalise(rows, data_dir, source="yfinance")


def _finalise(rows, data_dir, source="yfinance"):
    df = pd.DataFrame(rows)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "backtest_events.csv"), index=False)
    summary = summarise(df) if len(df) else {"n_events": 0}
    if len(df):
        summary["entry_timing"] = entry_summary(df)
        summary["exit_rules"] = exit_summary(df)
        summary["by_year"] = by_year_summary(df)
        summary["by_horizon"] = by_horizon_summary(df)
        summary["insider"] = insider_summary(df)
        if "sector" in df.columns and df["sector"].notna().any():
            summary["exit_rules_excl_avoided"] = exit_summary(df[~df["sector"].map(_avoid)])
            summary["exit_rules_favoured"] = exit_summary(df[df["sector"].map(_fav)])
            fav = df[df["sector"].map(_fav)]
            if "insider_signal" in fav.columns and fav["insider_signal"].notna().any():
                summary["insider_favoured"] = insider_summary(fav)
    if source == "eodhd":
        summary["caveats"] = [
            "Universe = EODHD incl. DELISTED names — SURVIVORSHIP-FREE.",
            "Fundamentals: EODHD, point-in-time by filing_date.",
            "Prices: EODHD adjusted close (splits & dividends handled).",
            "Insider = SEC Form 4 (US, point-in-time by filing date)."]
    else:
        summary["caveats"] = [
            "Universe = today's membership; delisted EXCLUDED (survivorship bias).",
            "Fundamentals: EODHD (point-in-time) if token set, else yfinance (flaky).",
            "AU 10y bond yield is a fixed proxy (%.1f%%)." % (AU_BOND_YIELD * 100)]
    with open(os.path.join(data_dir, "backtest_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {len(df)} events; summary saved")
    print(json.dumps({k: summary[k] for k in ("n_events", "baseline_3m", "by_horizon") if k in summary}, indent=2))


def run_eodhd(years, data_dir, limit=0, exchanges=("NYSE",), market="US"):
    """Survivorship-free backtest for one market: EODHD universe INCLUDING delisted
    names + EODHD adjusted EOD prices, streamed one ticker at a time (bounded memory).
    Fundamentals point-in-time (EODHD, all markets). Insider from SEC Form 4 is US-ONLY
    (SEC has no non-US filings), so the director-buy filter only applies to US runs.
    Benchmarks from yfinance indices (no survivorship issue). exchanges=None keeps all
    common stock (used for non-US markets, which have no NYSE-style venue families)."""
    import time
    from datetime import date, timedelta
    import requests
    if not eodhd.token():
        log("no EODHD_API_TOKEN — cannot run the survivorship-free path"); return
    if market == "US":
        load_us_bond()
    s = requests.Session()
    end = date.today()
    start = end - timedelta(days=int(years * 365) + 400)
    start_s, end_s = start.isoformat(), end.isoformat()
    mcfg = cfg.market_params(market)
    uni = eodhd.universe(market, include_delisted=True, exchanges=exchanges, session=s)
    if uni is None or len(uni) == 0:
        log("no EODHD universe, aborting"); return
    full_n = len(uni)
    if limit:
        uni = uni.head(limit)
        log(f"PROBE MODE: processing {len(uni)} of {full_n} tickers to measure per-ticker cost")
    benches = datamod.download_benchmarks(mcfg, period_days=int(years * 365) + 400)
    bench = benches.get(mcfg.BENCHMARK_300)
    if bench is None:
        bench = benches.get(mcfg.BENCHMARK_200)
    if bench is None:
        log("no benchmark, aborting"); return
    # yfinance daily bars may be tz-aware; EODHD prices are tz-naive. Align the
    # benchmark to a tz-naive daily index so index-relative drops are not all-NaN.
    bench = to_naive_daily(bench)
    log(f"benchmark {mcfg.BENCHMARK_300}: {len(bench)} points, "
        f"{bench.index.min().date()}..{bench.index.max().date()} (tz-naive)")
    name_by = dict(zip(uni["code"], uni["name"]))
    need = cfg.VOL_LOOKBACK + max(cfg.WINDOW_LENGTHS) + 5

    # Pass 1 — stream every ticker (incl. delisted); keep prices only for names with events.
    evmap, pxcache = {}, {}
    codes = list(uni["code"])
    eodhd.reset_quota_stats()
    t0 = time.time()
    scanned, overlap_logged = 0, False
    for i, code in enumerate(codes):
        px = eodhd.eod_series(code, market, start_s, end_s, session=s)
        time.sleep(0.02)
        # EODHD daily quota (100k calls, resets 00:00 UTC) exhausted -> every
        # remaining fetch will 402 too. Bail out NOW rather than spend 13 min
        # producing a near-empty dataset that then crashes downstream. Only judge
        # after a small sample so a couple of transient 402s don't trip it.
        n402, nok = eodhd.quota_stats()
        if n402 >= 20 and n402 > nok * 4:
            log(f"ABORT: EODHD daily quota exhausted — {n402} HTTP-402s vs {nok} OK "
                f"in the first {i+1} fetches. The 100k/day allowance resets at 00:00 UTC "
                f"(08:00 Perth). Re-run after the reset. No results written.")
            raise SystemExit(
                "EODHD daily API quota exhausted (HTTP 402). Wait for the 00:00 UTC "
                "reset (08:00 Perth) and re-run — this is not a code or data fault.")
        if px is None or len(px) < need:
            continue
        px = px[~px.index.duplicated(keep="last")].sort_index()
        px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
        scanned += 1
        if not overlap_logged:                       # one-time sanity check vs the tz bug
            ov = px.index.intersection(bench.index)
            log(f"index-overlap check ({code}): {len(ov)} shared dates of {len(px)} "
                f"price rows — {'OK' if len(ov) > 100 else 'WARNING: benchmark misaligned!'}")
            overlap_logged = True
        evs = detect_events(px["Close"], px["Volume"], bench, mcfg)
        if evs:
            evmap[code] = evs
            pxcache[code] = px
        if (i + 1) % 1000 == 0:
            per = (time.time() - t0) / (i + 1)
            log(f"prices {i+1}/{len(codes)} — {len(evmap)} names with events "
                f"({per*1000:.0f} ms/ticker, full-universe ETA {full_n*per/60:.0f} min)")
    elapsed = time.time() - t0
    per = elapsed / max(len(codes), 1)
    log(f"Pass 1 done: {len(codes)} tickers in {elapsed/60:.1f} min "
        f"({per*1000:.0f} ms/ticker), {scanned} had enough history to scan "
        f"({len(codes)-scanned} skipped). FULL-UNIVERSE ETA for Pass 1 = "
        f"{full_n*per/60:.0f} min over {full_n} tickers.")
    log(f"{market} survivorship-free: {sum(len(v) for v in evmap.values())} events across {len(evmap)} names")

    insider_map = {}
    if market == "US":                    # SEC Form 4 is US-only -> director filter is US-only
        try:
            ev_by_tk = {code: [e["date"] for e in evs] for code, evs in evmap.items()}
            insider_map = us_insiders.insider_signals_for_events(ev_by_tk, years=years)
        except Exception as e:  # noqa: BLE001
            log(f"insider fetch failed: {e!r}")
    else:
        log(f"{market}: no SEC insider data (director-buy filter is US-only) — gated model")

    # Pass 2 — fundamentals (point-in-time) + all metrics per event.
    rows = []
    for j, (code, evs) in enumerate(evmap.items()):
        px = pxcache[code]
        try:
            fund = eodhd.fundamentals(code, market, session=s)
        except Exception:  # noqa: BLE001
            fund = None
        sec = fund.get("sector") if fund else None
        for ev in evs:
            fm = forward_and_momentum(px["Close"], bench, ev)
            et = entry_timing_fields(px["Close"], px["Volume"], bench, ev)
            xr = exit_rules(px["Close"], ev)
            # confirmed-entry-anchored exits (trailing stop + non-recovery time-exit grid)
            cex = {}
            epos = find_entry(px["Close"], px["Volume"], ev["pos"], "confirmed")
            iw = ev["pos"] - ev["window_len"]
            if epos is not None and iw >= 0:
                cex = confirmed_exits(px["Close"], bench, epos, float(px["Close"].iloc[iw]))
            mm = metrics_for_event(fund, ev["date"], ev["entry"], market) if fund else {}
            ins = insider_map.get((code, ev["date"]), {})
            rows.append({"market": market, "ticker": code, "name": name_by.get(code, code),
                         "sector": sec, "exchange": f"{market}-SF", "date": ev["date"],
                         "year": int(ev["date"][:4]), "window_len": ev["window_len"],
                         "raw": ev["raw"], "z": ev["z"], "avg_daily_value": ev.get("avg_daily_value"),
                         **fm, **et, **xr, **cex, **mm, **ins})
        if (j + 1) % 1000 == 0:
            log(f"fundamentals {j+1}/{len(evmap)}")
    _finalise(rows, data_dir, source="eodhd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5)
    ap.add_argument("--markets", default="")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default="yfinance", choices=["yfinance", "eodhd"],
                    help="'eodhd' = survivorship-free US path (EODHD universe incl. delisted + prices)")
    ap.add_argument("--exchanges", default="NYSE",
                    help="eodhd path: canonical venue families to keep, comma-separated "
                         "(NYSE / NASDAQ / AMEX / ARCA). Default NYSE. 'ALL' keeps everything.")
    ap.add_argument("--market", default="US", choices=list(cfg.MARKETS),
                    help="eodhd path: which market to run survivorship-free (US/ASX/TSX/LSE).")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        from . import test_factor    # noqa: F401
        from . import test_insiders  # noqa: F401
        from . import test_universe  # noqa: F401
        from . import test_eodhd     # noqa: F401
        return
    if a.source == "eodhd":
        # Non-US markets have no NYSE-style venue families -> keep all common stock.
        if a.market == "US":
            exch = None if a.exchanges.strip().upper() == "ALL" else \
                [x.strip().upper() for x in a.exchanges.split(",") if x.strip()]
        else:
            exch = None
        run_eodhd(a.years, a.data_dir, a.limit, exchanges=exch, market=a.market)
    else:
        markets = [m.strip() for m in a.markets.split(",") if m.strip()] or list(cfg.MARKETS)
        run(markets, a.years, a.data_dir, a.limit)


if __name__ == "__main__":
    main()
