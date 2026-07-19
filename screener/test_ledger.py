"""Offline tests for the buy/sell ledger. No network. python -m screener.test_ledger"""
import os
import tempfile

import pandas as pd

from screener import ledger as L

passed = 0
def ok(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print("  ok ", name)


print("Date / return helpers:")
ok("add 3 months", L.add_months("2026-01-15", 3) == "2026-04-15")
ok("month-end clamp (Jan31 +3 -> Apr30)", L.add_months("2026-01-31", 3) == "2026-04-30")
ok("year rollover", L.add_months("2026-11-20", 3) == "2027-02-20")
ok("pct up 20%", abs(L.pct(10, 12) - 0.2) < 1e-9)
ok("pct guards zero entry", L.pct(0, 5) is None)

print("Due status:")
ok("past the mark -> SELL_DUE", L.due_status("2026-04-15", "2026-04-16") == "SELL_DUE")
ok("on the mark -> SELL_DUE", L.due_status("2026-04-15", "2026-04-15") == "SELL_DUE")
ok("within a week -> SELL_SOON", L.due_status("2026-04-15", "2026-04-10") == "SELL_SOON")
ok("far off -> OPEN", L.due_status("2026-04-15", "2026-01-10") == "OPEN")

print("Model ledger lifecycle:")
with tempfile.TemporaryDirectory() as dd:
    prices = {("US", "IBM"): 100.0, ("US", "PNR"): 50.0}
    look = lambda m, t: prices.get((m, t))
    buys = [{"market": "US", "ticker": "IBM", "name": "IBM", "sector": "Information Technology",
             "entry_date": "2026-01-15", "entry_price": 100.0},
            {"market": "US", "ticker": "PNR", "name": "Pentair", "sector": "Industrials",
             "entry_date": "2026-01-15", "entry_price": 50.0}]
    r1 = L.update_model_ledger(dd, buys, look, "2026-01-15")
    ok("two new model buys", len(r1["new_buys"]) == 2)
    ok("two open positions", r1["n_open"] == 2)
    ok("sell_due set to +3m", r1["open_positions"][0]["sell_due_date"] == "2026-04-15")
    ok("ledger file written", os.path.exists(os.path.join(dd, L.MODEL_LEDGER)))

    # same buys again next day -> no duplicates, still open, marked to market
    prices[("US", "IBM")] = 120.0     # IBM +20%
    r2 = L.update_model_ledger(dd, buys, look, "2026-01-16")
    ok("dedup: no new buys re-added", len(r2["new_buys"]) == 0)
    ibm = [p for p in r2["open_positions"] if p["ticker"] == "IBM"][0]
    ok("IBM marked to +20%", abs(ibm["ret_pct"] - 0.20) < 1e-9)

    # advance past the 3-month mark -> auto-closed, realised tally computed
    prices[("US", "IBM")] = 130.0; prices[("US", "PNR")] = 45.0
    r3 = L.update_model_ledger(dd, [], look, "2026-04-16")
    ok("both closed at the 3-month mark", len(r3["closed_this_run"]) == 2)
    ok("no open positions left", r3["n_open"] == 0)
    ok("realised tally counts 2", r3["realised_tally"]["n_closed"] == 2)
    ok("win rate 50% (IBM +30, PNR -10)", abs(r3["realised_tally"]["win_rate"] - 0.5) < 1e-9)
    led = pd.read_csv(os.path.join(dd, L.MODEL_LEDGER))
    ok("all rows CLOSED", (led["status"] == "CLOSED").all())

print("Personal holdings:")
with tempfile.TemporaryDirectory() as dd:
    pd.DataFrame([{"ticker": "AAPL", "market": "US", "entry_date": "2026-01-10",
                   "entry_price": 200.0, "quantity": 50, "notes": "core"}]).to_csv(
        os.path.join(dd, L.HOLDINGS), index=False)
    look = lambda m, t: 240.0 if t == "AAPL" else None
    h = L.holdings_status(dd, look, "2026-05-01")
    ok("one holding read", len(h) == 1)
    ok("marked to +20%", abs(h[0]["ret_pct"] - 0.20) < 1e-9)
    ok("past 3m -> SELL_DUE flag", h[0]["flag"] == "SELL_DUE")
    ok("missing file -> empty", L.holdings_status(tempfile.mkdtemp(), look, "2026-05-01") == [])

print(f"\nALL {passed} LEDGER ASSERTIONS PASSED")
