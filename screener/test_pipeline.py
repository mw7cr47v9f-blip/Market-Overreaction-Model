"""
Offline integration test of the orchestrator: monkeypatches the network layer
with synthetic data and checks the full stage1 -> stage2 -> dedup -> file-output
wiring, including that a second run correctly dedups. Run:
    python -m screener.test_pipeline
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

from screener import data as datamod
from screener import config as cfg
from screener import run as runmod


def _dates(n):
    return pd.bdate_range("2026-01-01", periods=n)

def _quiet(n, vol, start=10.0, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, vol, n); r[0] = 0
    return start * np.cumprod(1 + r)

def _crash_last(arr, pct):
    arr = arr.copy(); arr[-1] *= (1 + pct); return arr

N = 140
IDX = _dates(N)

def _df(close, vol_shares):
    return pd.DataFrame({"Close": close, "Volume": np.full(N, vol_shares)}, index=IDX)

UNIVERSE = pd.DataFrame({
    "code": ["AAA", "BIG", "SML", "CALM", "AVOI"],
    "name": ["Alpha Ltd", "Big Cap Ltd", "Small Ltd", "Calm Ltd", "Avoid Mining"],
    "yahoo": ["AAA.AX", "BIG.AX", "SML.AX", "CALM.AX", "AVOI.AX"],
    # sector drives the locked favoured-only alert filter
    "sector": ["Information Technology", "Industrials", "Software & Services",
               "Information Technology", "Materials"],
})
PRICES = {
    "AAA.AX":  _df(_crash_last(_quiet(N, 0.01, seed=1), -0.18), 5_000_000),   # mid cap crash
    "BIG.AX":  _df(_crash_last(_quiet(N, 0.01, seed=2), -0.22), 5_000_000),   # large cap crash (clears 20% floor)
    "SML.AX":  _df(_crash_last(_quiet(N, 0.01, seed=3), -0.18), 5_000_000),   # sub-$100m -> drop
    "CALM.AX": _df(_quiet(N, 0.01, seed=4), 5_000_000),                       # no crash
    "AVOI.AX": _df(_crash_last(_quiet(N, 0.01, seed=5), -0.24), 5_000_000),   # crash but AVOID sector (clears 20% floor)
}
BENCHES = {cfg.BENCHMARK_200: pd.Series(_quiet(N, 0.003, 7000, 99), index=IDX),
           cfg.BENCHMARK_300: pd.Series(_quiet(N, 0.003, 7000, 98), index=IDX)}
CAPS = {"AAA.AX": 800_000_000, "BIG.AX": 12_000_000_000, "SML.AX": 50_000_000,
        "AVOI.AX": 2_000_000_000}


def install(monkey):
    datamod.get_universe = lambda mcfg, local_fallback=None: UNIVERSE.copy()
    datamod.download_prices = lambda tickers, period_days=cfg.HISTORY_CALENDAR_DAYS: {
        k: v for k, v in PRICES.items() if k in tickers}
    datamod.download_benchmarks = lambda mcfg, period_days=cfg.HISTORY_CALENDAR_DAYS: BENCHES
    datamod.get_market_caps = lambda tickers: {k: v for k, v in CAPS.items() if k in tickers}
    runmod._fetch_announcements = lambda *a, **k: None   # no network in the test


passed = 0
def check(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok  {name}")


def run_once(data_dir):
    sys.argv = ["run", "--data-dir", data_dir, "--markets", "ASX"]
    runmod.main()


def main():
    install({})
    with tempfile.TemporaryDirectory() as d:
        print("First run:")
        run_once(d)
        new = json.load(open(os.path.join(d, "candidates_new.json")))
        tickers = {c["ticker"] for c in new["candidates"]}
        check("AAA (favoured mid-cap crash) alerted", "AAA" in tickers)
        check("BIG (favoured large-cap crash) alerted", "BIG" in tickers)
        check("SML (sub-$100m) NOT flagged", "SML" not in tickers)
        check("CALM (no crash) NOT flagged", "CALM" not in tickers)
        check("AVOI (avoid-sector crash) NOT alerted", "AVOI" not in tickers)
        big = next(c for c in new["candidates"] if c["ticker"] == "BIG")
        check("BIG benchmarked vs ASX200", big["benchmark"] == cfg.BENCHMARK_200)
        check("BIG tagged favoured + sector attached", big.get("favoured") is True and big.get("sector"))
        aaa = next(c for c in new["candidates"] if c["ticker"] == "AAA")
        check("AAA benchmarked vs ASX300", aaa["benchmark"] == cfg.BENCHMARK_300)
        allcsv = pd.read_csv(os.path.join(d, "candidates_all.csv"))
        check("candidates_all.csv has status=New", (allcsv["status"] == "New").all())
        check("AVOI logged to candidates_all (not alerted)", "AVOI" in set(allcsv["ticker"]))
        check("candidates_all carries sector column", "sector" in allcsv.columns)
        # all three size+price qualifiers (AAA, BIG, AVOI) are de-duped in state
        check("state.json records 3 seen keys", len(json.load(open(os.path.join(d, "state.json")))["seen"]) == 3)
        model = json.load(open(os.path.join(d, "ledger_status.json")))["model"]
        # Buy-on-the-drop model: both gated crashes (AAA, BIG) are bought the day they
        # qualify — no confirmation wait — so the ledger opens 2 today.
        check("ledger opens 2 on the drop (buy-on-drop: AAA + BIG)", model["n_open"] == 2)

        print("Second run (same data) — must dedup to zero new:")
        run_once(d)
        new2 = json.load(open(os.path.join(d, "candidates_new.json")))
        check("second run emits 0 new (dedup works)", new2["count"] == 0)
        allcsv2 = pd.read_csv(os.path.join(d, "candidates_all.csv"))
        check("candidates_all.csv unchanged on dedup run", len(allcsv2) == len(allcsv))

    print(f"\nALL {passed} PIPELINE ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
