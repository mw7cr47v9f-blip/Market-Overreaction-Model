"""
Confirmed-entry tracking for the LIVE screen — closes the "live gap" the docs flag.

A qualifying >=15% drop is NOT a buy. The model buys only on the *confirmed entry*:
within 15 trading days of the drop, the first day whose close breaks above the higher
of the previous two closes, on volume above its 20-day average. If no such day appears
inside 15 trading days, the trade is skipped entirely. Signals that never confirm are
the worst of the lot, so waiting filters out the falling knives.

So each gated drop enters a PENDING window. Every daily run re-checks the pending names
against fresh prices and moves each to one of:

    CONFIRMED  -> the breakout fired: BUY today (this is the entry the ledger records)
    PENDING    -> still inside the 15-day window, no breakout yet (day N of 15)
    EXPIRED    -> 15 trading days passed with no breakout: skipped, never bought

State persists in data/pending_confirmations.json; the day's outcome is written to
data/confirmations.json for the dashboard/email. The confirmation rule is the SAME
`find_entry(..., "confirmed")` used by the backtest, so live matches the tested model.
"""
from __future__ import annotations
import json
import os

import pandas as pd

from .backtest_factor import find_entry, ENTRY_MAX_WAIT

PENDING_JSON = "pending_confirmations.json"
CONF_JSON = "confirmations.json"


def _key(rec) -> str:
    return f"{rec.get('market')}|{rec.get('ticker')}|{rec.get('drop_date')}"


def load_pending(data_dir: str) -> list:
    try:
        with open(os.path.join(data_dir, PENDING_JSON)) as f:
            return json.load(f).get("pending", [])
    except Exception:  # noqa: BLE001
        return []


def save_pending(data_dir: str, pending: list, scan_date):
    with open(os.path.join(data_dir, PENDING_JSON), "w") as f:
        json.dump({"scan_date": scan_date, "pending": pending}, f, indent=2)


def write_confirmations(data_dir, confirmed, pending, expired, scan_date):
    with open(os.path.join(data_dir, CONF_JSON), "w") as f:
        json.dump({"scan_date": scan_date, "confirmed_today": confirmed,
                   "pending": pending, "expired": expired}, f, indent=2)


def check(close, volume, drop_date, max_wait: int = ENTRY_MAX_WAIT):
    """Return one of:
        ('confirmed', entry_date_iso, entry_price)  breakout fired within the window
        ('pending', n_trading_days_since_drop)       still waiting, inside the window
        ('expired', n_trading_days_since_drop)       window elapsed with no breakout
    Uses the backtest's `find_entry(..., 'confirmed')` so live == tested rule."""
    close = close.dropna().sort_index()
    volume = volume.reindex(close.index)
    idx = close.index
    if len(idx) < 3:
        return ("pending", 0)
    dd = pd.Timestamp(drop_date)
    before = [k for k, t in enumerate(idx) if t <= dd]
    if not before:
        return ("pending", 0)
    i = before[-1]                                  # position of the drop bar
    n_since = (len(idx) - 1) - i                    # trading bars observed since the drop
    j = find_entry(close, volume, i, "confirmed", max_wait=max_wait)
    if j is not None and j < len(idx):
        return ("confirmed", str(pd.Timestamp(idx[j]).date()), round(float(close.iloc[j]), 4))
    if n_since >= max_wait:
        return ("expired", n_since)
    return ("pending", n_since)


def update(data_dir, new_drops, series_for, scan_date):
    """Advance the confirmation state machine one trading day.

    new_drops:  today's gated drops (dicts with market, ticker, name, sector,
                event_date/drop_date, last_close).
    series_for(rec) -> (close_series, volume_series) or None for a pending record.

    Returns (confirmed_today, still_pending, expired_today) and persists state.
    Confirmed records carry entry_date + entry_price (the breakout bar)."""
    by_key = {_key(p): p for p in load_pending(data_dir)}
    for d in new_drops:
        rec = {"market": d.get("market"), "ticker": d.get("ticker"),
               "name": d.get("name"), "sector": d.get("sector"),
               "drop_date": d.get("event_date") or d.get("drop_date") or scan_date,
               "drop_price": d.get("last_close")}
        by_key.setdefault(_key(rec), rec)

    confirmed, still, expired = [], [], []
    for rec in by_key.values():
        cv = None
        try:
            cv = series_for(rec)
        except Exception:  # noqa: BLE001
            cv = None
        if cv is None:
            # No fresh price series (e.g. dropped out of the universe). Age it by run
            # count so a stuck record still expires rather than lingering forever.
            rec["day"] = int(rec.get("day", 0)) + 1
            (expired if rec["day"] >= ENTRY_MAX_WAIT else still).append(rec)
            continue
        close, volume = cv
        status = check(close, volume, rec["drop_date"])
        if status[0] == "confirmed":
            rec["entry_date"], rec["entry_price"] = status[1], status[2]
            confirmed.append(rec)
        elif status[0] == "expired":
            rec["day"] = status[1]
            expired.append(rec)
        else:
            rec["day"] = status[1]
            still.append(rec)

    save_pending(data_dir, still, scan_date)
    write_confirmations(data_dir, confirmed, still, expired, scan_date)
    return confirmed, still, expired
