"""Offline logic tests for the SEC Form-4 insider signal. No network.
Run: python -m screener.test_insiders"""
import pandas as pd

from screener.us_insiders import parse_form4_xml, classify_window, net_signal
from screener.backtest_factor import insider_summary

passed = 0
def ok(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print("  ok ", name)


# A minimal but realistic Form-4: a DIRECTOR buys 1,000 @ $10 (P), then sells
# 500 @ $12 (S). Namespaced-free, values wrapped in <value> like the real feed.
XML = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>0</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-01-15</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>10.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-02-01</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>12.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

# A grant-only Form-4 (code A) — must NOT count as a discretionary buy signal.
XML_GRANT = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner><reportingOwnerRelationship><isOfficer>1</isOfficer>
  </reportingOwnerRelationship></reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <transactionDate><value>2024-01-20</value></transactionDate>
    <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
    <transactionAmounts>
      <transactionShares><value>2000</value></transactionShares>
      <transactionPricePerShare><value>0</value></transactionPricePerShare>
      <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
    </transactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""

print("Form-4 XML parsing:")
p = parse_form4_xml(XML)
ok("director flag read", p["is_director"] is True)
ok("two transactions parsed", len(p["transactions"]) == 2)
ok("first is a purchase P", p["transactions"][0]["code"] == "P")
ok("buy value = 1000*10 = 10000", abs(p["transactions"][0]["value"] - 10000) < 1e-6)
ok("second is a sale S", p["transactions"][1]["code"] == "S")
ok("sell value = 500*12 = 6000", abs(p["transactions"][1]["value"] - 6000) < 1e-6)
ok("transaction date parsed", p["transactions"][0]["date"] == "2024-01-15")
ok("malformed xml -> empty, no crash", parse_form4_xml("<notxml>")["transactions"] == [])

print("Windowing / classification:")
filing = {"filing_date": "2024-02-02", "parsed": p}
agg = classify_window([filing], "2024-03-01")   # both trades within prior 182d
ok("one buy counted", agg["insider_buy_n"] == 1)
ok("one sell counted", agg["insider_sell_n"] == 1)
ok("net = 10000 - 6000 = 4000", abs(agg["insider_net_val"] - 4000) < 1e-6)
ok("director_buy flagged", agg["director_buy"] is True)
ok("net-positive -> 'buy' signal", agg["insider_signal"] == "buy")

grant = {"filing_date": "2024-01-21", "parsed": parse_form4_xml(XML_GRANT)}
gagg = classify_window([grant], "2024-03-01")
ok("grant (code A) is NOT a buy", gagg["insider_buy_n"] == 0)
ok("grant-only -> 'none' signal", gagg["insider_signal"] == "none")

# transaction BEFORE the lookback window is excluded
old_agg = classify_window([filing], "2025-06-01")  # >182d after Feb 2024 trades
ok("stale trades fall out of the window", old_agg["insider_buy_n"] == 0)

# a filing that only became public AFTER the event must not leak in
late = {"filing_date": "2024-03-10", "parsed": p}
late_agg = classify_window([late], "2024-03-01")
ok("late-filed report excluded (no lookahead)", late_agg["insider_buy_n"] == 0)

print("net_signal edge cases:")
ok("no trades -> none", net_signal(0, 0, 0.0) == "none")
ok("net positive buy -> buy", net_signal(2, 0, 5000.0) == "buy")
ok("net negative sell -> sell", net_signal(0, 3, -5000.0) == "sell")
ok("offsetting -> mixed", net_signal(1, 1, 0.0) == "mixed")

print("insider_summary bucketing:")
rows = []
for i in range(14):
    dbuy = i % 2 == 0
    rows.append({"market": "US", "insider_signal": "buy" if dbuy else "sell",
                 "director_buy": dbuy, "director_sell": not dbuy,
                 "fwd_3m": (0.10 if dbuy else -0.02) + 0.001 * i,
                 "excess_3m": (0.08 if dbuy else -0.03) + 0.001 * i,
                 "fwd_6m": (0.15 if dbuy else 0.0) + 0.001 * i,
                 "excess_6m": (0.10 if dbuy else -0.02) + 0.001 * i})
s = insider_summary(pd.DataFrame(rows))
ok("signal count present", s["n_with_signal"] == 14)
ok("buy bucket present", "buy" in s["by_signal"] and s["by_signal"]["buy"]["n"] == 7)
ok("director_buy split present", "yes" in s["director_buy"] and "no" in s["director_buy"])
ok("director-buy beats director-sell (3m)",
   s["director_buy"]["yes"]["3m"]["mean"] > s["director_buy"]["no"]["3m"]["mean"])
ok("doubling metric computed",
   s["double_on_director_buy_3m"]["double_dirbuy"] > s["double_on_director_buy_3m"]["equal_weight"])

print(f"\nALL {passed} INSIDER ASSERTIONS PASSED")
