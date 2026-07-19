"""
Two-sided BUY / SELL ledger, maintained by the GitHub Actions screener (which
has repo write access; the Claude analysis task is read-only, so it can't persist
here — it just reads the outputs and reports them).

Two books:
  * MODEL ledger (data/model_ledger.csv) — automatic. Every favoured-sector
    overreaction signal becomes a model position: entry at the signal close, a
    fixed 3-month hold (the locked exit rule), marked-to-market daily and closed
    automatically when the 3 months are up. Gives a running, hindsight-free tally
    of what the model itself would have done.
  * PERSONAL holdings (data/holdings.csv) — you maintain this file (ticker,
    market, entry_date, entry_price, quantity, notes). The screener marks it to
    market and flags anything past its 3-month mark so the daily feed prompts the
    sell alongside the buys.

Pure helpers (add_months, pct, due_status) are unit-tested offline; only the
CSV read/write touches disk.
"""
from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd

from . import config as cfg

MODEL_LEDGER = "model_ledger.csv"
HOLDINGS = "holdings.csv"
STATUS_JSON = "ledger_status.json"
SELL_SOON_DAYS = 7          # flag positions coming due within a week


# ---- pure helpers (offline-tested) ---------------------------------------

def add_months(iso: str, months: int) -> str:
    d = date.fromisoformat(str(iso)[:10])
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    # clamp day to the end of the target month
    import calendar
    day = min(d.day, calendar.monthrange(y, m)[1])
    return date(y, m, day).isoformat()


def pct(entry, last) -> float | None:
    try:
        entry = float(entry); last = float(last)
        if entry <= 0:
            return None
        return round(last / entry - 1.0, 4)
    except (TypeError, ValueError):
        return None


def due_status(sell_due_iso: str, scan_iso: str) -> str:
    """OPEN, SELL_DUE (past the 3-month mark) or SELL_SOON (within a week)."""
    due = date.fromisoformat(str(sell_due_iso)[:10])
    now = date.fromisoformat(str(scan_iso)[:10])
    if now >= due:
        return "SELL_DUE"
    if (due - now).days <= SELL_SOON_DAYS:
        return "SELL_SOON"
    return "OPEN"


def _key(market, ticker, entry_date):
    return f"{market}:{ticker}:{entry_date}"


# ---- MODEL ledger ---------------------------------------------------------

_COLS = ["market", "ticker", "name", "sector", "entry_date", "entry_price",
         "sell_due_date", "status", "last_price", "ret_pct", "days_held",
         "exit_date", "exit_price"]


def update_model_ledger(data_dir, new_buys, price_lookup, scan_date) -> dict:
    """new_buys: [{market,ticker,name,sector,entry_date,entry_price}].
    price_lookup: (market,ticker) -> last price or None. Returns a report dict."""
    path = os.path.join(data_dir, MODEL_LEDGER)
    led = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=_COLS)
    for c in _COLS:
        if c not in led.columns:
            led[c] = None
    # string/date columns must be object dtype, else assigning e.g. an exit_date
    # into an all-None (float64) column raises in modern pandas.
    for c in ("market", "ticker", "name", "sector", "entry_date", "sell_due_date",
              "status", "exit_date"):
        led[c] = led[c].astype(object)
    have = {_key(r.market, r.ticker, r.entry_date) for r in led.itertuples()}

    # 1) add genuinely-new model buys (favoured only — caller filters)
    added = []
    for b in new_buys:
        k = _key(b["market"], b["ticker"], b["entry_date"])
        if k in have:
            continue
        row = {**b, "sell_due_date": add_months(b["entry_date"], cfg.HOLD_MONTHS),
               "status": "OPEN", "last_price": b["entry_price"], "ret_pct": 0.0,
               "days_held": 0, "exit_date": None, "exit_price": None}
        led = pd.concat([led, pd.DataFrame([row])], ignore_index=True)
        have.add(k); added.append(row)

    # 2) mark-to-market + auto-close anything past its 3-month mark
    closed_now, due, soon, open_rows = [], [], [], []
    for i, r in led.iterrows():
        if r["status"] == "CLOSED":
            continue
        last = price_lookup(r["market"], r["ticker"])
        if last is not None:
            led.at[i, "last_price"] = round(float(last), 4)
            led.at[i, "ret_pct"] = pct(r["entry_price"], last)
        led.at[i, "days_held"] = (date.fromisoformat(str(scan_date)[:10])
                                  - date.fromisoformat(str(r["entry_date"])[:10])).days
        st = due_status(r["sell_due_date"], scan_date)
        if st == "SELL_DUE":
            led.at[i, "status"] = "CLOSED"
            led.at[i, "exit_date"] = scan_date
            led.at[i, "exit_price"] = led.at[i, "last_price"]
            closed_now.append(_rowdict(led.loc[i]))
        else:
            led.at[i, "status"] = "OPEN"
            (soon if st == "SELL_SOON" else open_rows).append(_rowdict(led.loc[i]))

    led.to_csv(path, index=False)

    closed = led[led["status"] == "CLOSED"]
    realised = closed["ret_pct"].dropna().astype(float)
    tally = {"n_closed": int(len(realised)),
             "win_rate": round(float((realised > 0).mean()), 3) if len(realised) else None,
             "avg_return": round(float(realised.mean()), 4) if len(realised) else None,
             "median_return": round(float(realised.median()), 4) if len(realised) else None}
    return {"scan_date": scan_date, "new_buys": added, "closed_this_run": closed_now,
            "sell_soon": soon, "open_positions": open_rows,
            "n_open": len(open_rows) + len(soon), "realised_tally": tally}


def _rowdict(r):
    return {"market": r["market"], "ticker": r["ticker"], "name": r["name"],
            "sector": r.get("sector"), "entry_date": r["entry_date"],
            "entry_price": _num(r["entry_price"]), "last_price": _num(r["last_price"]),
            "ret_pct": _num(r["ret_pct"]), "days_held": _num(r["days_held"]),
            "sell_due_date": r["sell_due_date"]}


def _num(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


# ---- PERSONAL holdings ----------------------------------------------------

def holdings_status(data_dir, price_lookup, scan_date) -> list:
    """Read the user-maintained holdings.csv, mark to market, flag 3-month sells."""
    path = os.path.join(data_dir, HOLDINGS)
    if not os.path.exists(path):
        return []
    try:
        h = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return []
    h.columns = [c.strip().lower() for c in h.columns]
    out = []
    for r in h.itertuples():
        tk = str(getattr(r, "ticker", "")).upper().strip()
        if not tk:
            continue
        mk = str(getattr(r, "market", "")).upper().strip() or "US"
        entry_price = getattr(r, "entry_price", None)
        entry_date = str(getattr(r, "entry_date", "") or "")[:10]
        last = price_lookup(mk, tk)
        rec = {"market": mk, "ticker": tk, "entry_date": entry_date or None,
               "entry_price": _num(entry_price), "quantity": _num(getattr(r, "quantity", None)),
               "last_price": _num(last), "ret_pct": pct(entry_price, last) if last else None,
               "notes": (str(getattr(r, "notes", "")) if hasattr(r, "notes") else "")}
        if entry_date:
            rec["sell_due_date"] = add_months(entry_date, cfg.HOLD_MONTHS)
            rec["flag"] = due_status(rec["sell_due_date"], scan_date)
        else:
            rec["flag"] = "NO_DATE"
        out.append(rec)
    return out


def write_status(data_dir, model_report, holdings):
    with open(os.path.join(data_dir, STATUS_JSON), "w") as f:
        json.dump({"model": model_report, "holdings": holdings}, f, indent=2)
