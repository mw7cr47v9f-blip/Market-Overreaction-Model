"""Offline tests for the EODHD fundamentals parsing + point-in-time metrics.
No network. python -m screener.test_eodhd"""
from screener import eodhd
from screener.backtest_factor import metrics_for_event

passed = 0
def ok(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print("  ok ", name)


# A minimal but realistic EODHD /fundamentals payload: two fiscal years, with
# filing_date stamped AFTER each fiscal-period end (the point-in-time key).
JS = {
    "General": {"Code": "TESTCO", "Sector": "Technology", "Industry": "Software"},
    "SharesStats": {"SharesOutstanding": 100},
    "outstandingShares": {"annual": {
        "0": {"dateFormatted": "2024-06-30", "shares": 100},
        "1": {"dateFormatted": "2023-06-30", "shares": 100}}},
    "Financials": {
        "Income_Statement": {"yearly": {
            "2023-06-30": {"date": "2023-06-30", "filing_date": "2023-08-15", "totalRevenue": 1000, "netIncome": 100},
            "2024-06-30": {"date": "2024-06-30", "filing_date": "2024-08-15", "totalRevenue": 1200, "netIncome": 150}}},
        "Balance_Sheet": {"yearly": {
            "2023-06-30": {"filing_date": "2023-08-15", "totalStockholderEquity": 500, "totalDebt": 100},
            "2024-06-30": {"filing_date": "2024-08-15", "totalStockholderEquity": 600, "totalDebt": 120}}},
        "Cash_Flow": {"yearly": {
            "2023-06-30": {"freeCashFlow": 80},
            "2024-06-30": {"freeCashFlow": 120}}},
    },
}

print("Parse EODHD fundamentals:")
fund = eodhd.parse_fundamentals(JS)
ok("two fiscal years parsed", len(fund["fy"]) == 2)
ok("sector read", fund["sector"] == "Technology")
ok("ordered oldest-first", fund["fy"][0]["date"] == "2023-06-30")
ok("filing_date carried", fund["fy"][1]["filing_date"] == "2024-08-15")
ok("net income read", fund["fy"][1]["ni"] == 150)
ok("free cash flow read", fund["fy"][1]["fcf"] == 120)
ok("eps computed = ni/shares", abs(fund["fy"][1]["eps"] - 1.5) < 1e-9)

print("Point-in-time metrics:")
# Event AFTER the FY2024 filing (Aug 2024) -> uses FY2024
m = metrics_for_event(fund, "2024-10-01", 50.0, "US")
ok("uses latest filed FY (2024)", m["npat_margin"] == round(150/1200, 4))     # 12.5%
ok("fcf margin", m["fcf_margin"] == round(120/1200, 4))                        # 10%
ok("roe", m["roe"] == round(150/600, 4))
ok("rev growth CAGR 1000->1200", m["rev_growth"] == round(1200/1000 - 1, 4))
ok("eps growth CAGR 1.0->1.5", m["eps_growth"] == round(1.5/1.0 - 1, 4))

# Event BETWEEN the two filings (say 2024-07-01, before the Aug-2024 filing) ->
# must NOT see FY2024 yet; falls back to FY2023 (no lookahead).
m2 = metrics_for_event(fund, "2024-07-01", 50.0, "US")
ok("no lookahead: uses FY2023 before FY2024 is filed", m2["npat_margin"] == round(100/1000, 4))  # 10%
ok("no growth yet (only one filed FY)", m2["rev_growth"] is None)

print("Dividend yield (EODHD list form):")
fund["dividends"] = [("2024-03-15", 0.5), ("2024-09-15", 0.5), ("2023-01-01", 9.0)]
m3 = metrics_for_event(fund, "2024-10-01", 50.0, "US")
ok("TTM dividends 0.5+0.5 over price 50 = 2%", m3["div_yield"] == round(1.0/50, 4))

print("Guards:")
ok("empty payload -> None", eodhd.parse_fundamentals({}) is None)
ok("no fy -> {}", metrics_for_event({"fy": []}, "2024-01-01", 10.0, "US") == {})

print("Symbol-list + EOD parsing:")
syms = eodhd.parse_symbol_list([
    {"Code": "AAA", "Name": "Alpha", "Type": "Common Stock", "Exchange": "NYSE", "Currency": "USD"},
    {"Code": "ETF1", "Name": "Some ETF", "Type": "ETF"},
    {"Code": "DEAD", "Name": "Dead Co", "Type": "Common Stock"}])   # delisted common stock kept
ok("common stock kept, ETF dropped", {r["code"] for r in syms} == {"AAA", "DEAD"})
ok("NYSE not OTC", eodhd._is_otc("NYSE") is False)
ok("NASDAQ not OTC", eodhd._is_otc("NASDAQ") is False)
ok("OTCMKTS flagged OTC", eodhd._is_otc("OTCMKTS") is True)
ok("PINK flagged OTC", eodhd._is_otc("PINK") is True)
ok("GREY flagged OTC", eodhd._is_otc("OTC GREY") is True)
ok("None venue not OTC", eodhd._is_otc(None) is False)
ok("family NYSE", eodhd._venue_family("NYSE") == "NYSE")
ok("family NYSE MKT -> AMEX (not NYSE)", eodhd._venue_family("NYSE MKT") == "AMEX")
ok("family NYSE American -> AMEX", eodhd._venue_family("NYSE American") == "AMEX")
ok("family NYSE ARCA -> ARCA (not NYSE)", eodhd._venue_family("NYSE ARCA") == "ARCA")
ok("family NASDAQ", eodhd._venue_family("NASDAQ") == "NASDAQ")
ok("family NASDAQ Global Select -> NASDAQ", eodhd._venue_family("NASDAQ Global Select") == "NASDAQ")
ok("family OTCMKTS -> OTC", eodhd._venue_family("OTCMKTS") == "OTC")
ok("family empty -> (none)", eodhd._venue_family("") == "(none)")
px = eodhd.parse_eod([
    {"date": "2024-01-02", "close": 10, "adjusted_close": 9.5, "volume": 1000},
    {"date": "2024-01-03", "close": 11, "adjusted_close": 10.4, "volume": 1200}])
ok("eod parsed to 2 rows", len(px) == 2)
ok("uses adjusted_close", abs(px["Close"].iloc[0] - 9.5) < 1e-9)
ok("empty eod -> None", eodhd.parse_eod([]) is None)

print("Streaming survivorship-free run (monkeypatched, no network):")
import os, tempfile, json
import numpy as np, pandas as pd
from screener import backtest_factor as bf
from screener import data as datamod
from screener import us_insiders
from screener import config as cfg

_N = 400
_idx = pd.bdate_range("2023-01-02", periods=_N)
def _crash_series():
    rng = np.random.default_rng(0); r = rng.normal(0, 0.008, _N); r[0] = 0
    c = 10 * np.cumprod(1 + r); c[150] = c[149] * 0.82           # -18% crash
    c[151] = c[150] * 1.03                                        # next-day up-tick -> confirmed entry
    for i in range(152, _N): c[i] = c[i-1] * 1.001                # slow recovery, full fwd window
    return pd.DataFrame({"Close": c, "Volume": np.full(_N, 1e8)}, index=_idx)
def _calm_series():
    rng = np.random.default_rng(1); r = rng.normal(0, 0.008, _N); r[0] = 0
    return pd.DataFrame({"Close": 10*np.cumprod(1+r), "Volume": np.full(_N, 1e8)}, index=_idx)
_PX = {"AAA": _crash_series(), "CALM": _calm_series()}
_bench = pd.Series(np.full(_N, 7000.0), index=_idx)

bf.eodhd.token = lambda: "TESTTOKEN"
bf.eodhd.universe = lambda market="US", include_delisted=True, exchanges=("NYSE",), session=None: pd.DataFrame(
    {"code": ["AAA", "CALM"], "name": ["Alpha", "Calm"]})
bf.eodhd.eod_series = lambda code, market, start, end, session=None: _PX.get(code)
bf.eodhd.fundamentals = lambda code, market, session=None: {
    "fy": [{"date": "2022-06-30", "filing_date": "2022-08-15", "ni": 100, "eq": 500, "debt": 100,
            "rev": 1000, "eps": 1.0, "fcf": 80}],
    "shares_out": 100, "sector": "Information Technology", "dividends": None}
datamod.download_benchmarks = lambda mcfg, period_days=0: {cfg.market_params("US").BENCHMARK_300: _bench}
us_insiders.insider_signals_for_events = lambda ev_by_tk, years=5: {}

with tempfile.TemporaryDirectory() as dd:
    bf.run_eodhd(years=1, data_dir=dd, limit=0)
    ev = pd.read_csv(os.path.join(dd, "backtest_events.csv"))
    summ = json.load(open(os.path.join(dd, "backtest_summary.json")))
    ok("AAA crash produced an event", "AAA" in set(ev["ticker"]))
    ok("CALM produced no event", "CALM" not in set(ev["ticker"]))
    ok("events tagged survivorship-free exchange", (ev["exchange"] == "US-SF").all())
    ok("point-in-time fundamentals attached", ev["npat_margin"].notna().any())
    ok("survivorship-free caveat written",
       any("SURVIVORSHIP-FREE" in c for c in summ.get("caveats", [])))

print(f"\nALL {passed} EODHD ASSERTIONS PASSED")
