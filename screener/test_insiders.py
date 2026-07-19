"""Offline logic tests for the SEC Form-4 insider signal. No network.
Run: python -m screener.test_insiders"""
import datetime as dt
import pandas as pd

from screener.us_insiders import (parse_form4_xml, parsed_from_bulk, classify_window,
                                  net_signal, _norm_date, quarters_between, _dataset_urls)
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

print("Bulk-table decoding (the live path):")
# NONDERIV_TRANS rows as they arrive from the quarterly TSV (DD-MON-YYYY dates,
# string numbers), plus REPORTINGOWNER relationship strings for the accession.
bulk_trans = [
    {"date": "15-JAN-2024", "code": "P", "shares": "1000", "price": "10.0", "acq": "A"},
    {"date": "01-FEB-2024", "code": "S", "shares": "500", "price": "12.0", "acq": "D"},
    {"date": "20-JAN-2024", "code": "M", "shares": "300", "price": "0", "acq": "A"},  # option exercise, ignored
]
bp = parsed_from_bulk(["DIRECTOR", "OFFICER"], bulk_trans)
ok("bulk director flag from relationship string", bp["is_director"] is True)
ok("bulk parses 3 raw transactions", len(bp["transactions"]) == 3)
ok("bulk DD-MON-YYYY date -> ISO", bp["transactions"][0]["date"] == "2024-01-15")
ok("bulk buy value 1000*10", abs(bp["transactions"][0]["value"] - 10000) < 1e-6)
bfil = {"filing_date": "2024-02-05", "parsed": bp}
bagg = classify_window([bfil], "2024-03-01")
ok("bulk buy counted", bagg["insider_buy_n"] == 1)
ok("bulk sell counted", bagg["insider_sell_n"] == 1)
ok("bulk option-exercise (M) NOT counted as buy/sell",
   bagg["insider_buy_n"] == 1 and bagg["insider_sell_n"] == 1)
ok("bulk net = 4000 -> buy", bagg["insider_signal"] == "buy" and abs(bagg["insider_net_val"] - 4000) < 1e-6)

print("Date normalisation:")
ok("ISO passes through", _norm_date("2024-01-15") == "2024-01-15")
ok("DD-MON-YYYY", _norm_date("02-JAN-2024") == "2024-01-02")
ok("YYYYMMDD", _norm_date("20240115") == "2024-01-15")
ok("junk -> None", _norm_date("not-a-date") is None)

print("Quarter maths / dataset URLs:")
qs = quarters_between(dt.date(2023, 11, 1), dt.date(2024, 5, 15))
ok("spans 2023Q4..2024Q2", qs[0] == (2023, 4) and qs[-1] == (2024, 2) and len(qs) == 3)
ok("single-quarter range", quarters_between(dt.date(2024, 2, 1), dt.date(2024, 2, 28)) == [(2024, 1)])
urls = _dataset_urls(2024, 1)
ok("dataset url names the quarter zip", any("2024q1_form345.zip" in u for u in urls))
ok("two host fallbacks offered", len(urls) == 2)

print("Bulk zip indexing (issuer filter + 3-table join):")
import io as _io, zipfile as _zip
from screener.us_insiders import _index_quarter
_SUB = ("ACCESSION_NUMBER\tISSUERCIK\tISSUERTRADINGSYMBOL\tFILING_DATE\n"
        "0001-24-000001\t320193\tAAPL\t17-JAN-2024\n"
        "0001-24-000002\t320193\tAAPL\t03-FEB-2024\n"
        "0009-24-000009\t999999\tZZZZ\t10-JAN-2024\n")
_OWN = ("ACCESSION_NUMBER\tRPTOWNER_RELATIONSHIP\n"
        "0001-24-000001\tDIRECTOR\n0001-24-000002\tOFFICER\n0009-24-000009\tDIRECTOR\n")
_TRA = ("ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
        "0001-24-000001\t15-JAN-2024\tP\t1000\t10.0\tA\n"
        "0001-24-000002\t01-FEB-2024\tS\t500\t12.0\tD\n"
        "0009-24-000009\t05-JAN-2024\tP\t9999\t50.0\tA\n")
_buf = _io.BytesIO()
with _zip.ZipFile(_buf, "w") as _z:
    _z.writestr("SUBMISSION.tsv", _SUB)
    _z.writestr("REPORTINGOWNER.tsv", _OWN)
    _z.writestr("NONDERIV_TRANS.tsv", _TRA)
_buf.seek(0)
_idx = _index_quarter(_zip.ZipFile(_buf), {320193: "AAPL"})
ok("only wanted issuer indexed (ZZZZ filtered)", list(_idx.keys()) == ["AAPL"])
ok("both AAPL accessions joined", len(_idx["AAPL"]) == 2)
_bagg = classify_window(_idx["AAPL"], "2024-03-01")
ok("indexed buy/sell counted", _bagg["insider_buy_n"] == 1 and _bagg["insider_sell_n"] == 1)
ok("indexed net = 4000 -> buy", abs(_bagg["insider_net_val"] - 4000) < 1e-6 and _bagg["insider_signal"] == "buy")
ok("indexed director_buy flagged", _bagg["director_buy"] is True)

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
