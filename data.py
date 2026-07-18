"""
Persistent dedup state + the accumulating candidate log.

state.json  -> {"seen": ["TICKER:YYYY-MM-DD", ...], "last_scan": "YYYY-MM-DD"}
              one entry per (ticker, event_date). A crash is logged ONCE even
              though the rolling 5-day window keeps containing it for days.
candidates_all.csv -> every candidate ever emitted (append-only feed for Claude).
candidates_new.json -> just this run's genuinely-new candidates (the alert).
"""
from __future__ import annotations

import json
import os
from typing import Iterable

import pandas as pd


def load_seen(state_path: str) -> set[str]:
    if not os.path.exists(state_path):
        return set()
    with open(state_path) as f:
        return set(json.load(f).get("seen", []))


def save_state(state_path: str, seen: Iterable[str], last_scan: str):
    with open(state_path, "w") as f:
        json.dump({"seen": sorted(set(seen)), "last_scan": last_scan}, f, indent=2)


def append_candidates(csv_path: str, rows: list[dict]):
    if not rows:
        return
    new = pd.DataFrame(rows)
    # status is filled in later by the Claude analysis step; seed as "New".
    if "status" not in new.columns:
        new.insert(0, "status", "New")
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined.to_csv(csv_path, index=False)


def write_new(json_path: str, rows: list[dict], scan_date: str):
    with open(json_path, "w") as f:
        json.dump({"scan_date": scan_date, "count": len(rows), "candidates": rows},
                  f, indent=2)
