"""Offline tests for the US expanded (S&P 1500 + Nasdaq) universe. No network.
python -m screener.test_universe"""
import pandas as pd

from screener import data as datamod

passed = 0
def ok(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print("  ok ", name)


print("Market-cap parsing:")
ok("$1.2bn string", datamod._parse_cap("$1,200,000,000") == 1_200_000_000.0)
ok("plain number", datamod._parse_cap(2_500_000_000) == 2_500_000_000.0)
ok("N/A -> None", datamod._parse_cap("N/A") is None)
ok("blank -> None", datamod._parse_cap("") is None)

print("S&P 1500 + Nasdaq union / dedup:")
# stub the two network fetchers with synthetic frames
sp = pd.DataFrame({"code": ["AAPL", "MSFT", "IBM"], "name": ["Apple", "Microsoft", "IBM"],
                   "sector": ["Information Technology", "Information Technology", "Information Technology"],
                   "yahoo": ["AAPL", "MSFT", "IBM"], "exchange": ["SP1500"] * 3})
nq = pd.DataFrame({"code": ["AAPL", "FAKE", "GROW"],  # AAPL overlaps S&P; FAKE/GROW are new
                   "name": ["Apple", "Fake Co", "Grow Inc"],
                   "sector": ["Information Technology", "Health Care", "Software & Services"],
                   "yahoo": ["AAPL", "FAKE", "GROW"], "exchange": ["NASDAQ"] * 3})
datamod._sp1500_universe = lambda fb: sp.copy()
datamod._nasdaq_universe = lambda mc: nq.copy()

out = datamod._us_expanded_universe(None)
ok("union size = 3 S&P + 2 new Nasdaq = 5", len(out) == 5)
ok("AAPL kept once (S&P wins the dedup)", (out["code"] == "AAPL").sum() == 1)
ok("AAPL tagged SP1500 not NASDAQ", out.loc[out.code == "AAPL", "exchange"].iloc[0] == "SP1500")
ok("new Nasdaq names present", set(["FAKE", "GROW"]).issubset(set(out["code"])))
ok("Nasdaq-only tagged NASDAQ", out.loc[out.code == "GROW", "exchange"].iloc[0] == "NASDAQ")
ok("sector carried through", out.loc[out.code == "GROW", "sector"].iloc[0] == "Software & Services")

print("Fallback when Nasdaq fails:")
def _boom(mc):
    raise RuntimeError("nasdaq down")
datamod._nasdaq_universe = _boom
out2 = datamod._us_expanded_universe(None)
ok("falls back to S&P 1500 alone", len(out2) == 3 and set(out2["code"]) == {"AAPL", "MSFT", "IBM"})

print(f"\nALL {passed} UNIVERSE ASSERTIONS PASSED")
