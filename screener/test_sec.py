"""Offline tests for SEC companyfacts -> margins. python -m screener.test_sec"""
from screener import sec_fundamentals as secf
from screener import config as cfg

passed = 0
def ok(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print("  ok ", name)


# Synthetic companyfacts: two annual 10-Ks; the gate should read the LATEST (2023).
JS = {
    "cik": 320193, "entityName": "TESTCO",
    "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"end": "2022-12-31", "val": 1000, "form": "10-K"},
            {"end": "2023-12-31", "val": 1200, "form": "10-K"},
            {"end": "2023-09-30", "val": 300,  "form": "10-Q"}]}},   # quarterly ignored
        "NetIncomeLoss": {"units": {"USD": [
            {"end": "2022-12-31", "val": 100, "form": "10-K"},
            {"end": "2023-12-31", "val": 180, "form": "10-K"}]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            {"end": "2023-12-31", "val": 240, "form": "10-K"}]}},
        "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [
            {"end": "2023-12-31", "val": 60, "form": "10-K"}]}},     # positive outflow
    }},
}

print("Parse SEC companyfacts -> margins:")
m = secf.margins_from_companyfacts(JS)
ok("npat margin = latest NI/Rev (180/1200)", m["npat_margin"] == round(180/1200, 4))   # 0.15
ok("fcf margin = (OCF-capex)/Rev ((240-60)/1200)", m["fcf_margin"] == round(180/1200, 4))  # 0.15
ok("uses latest annual (2023 not 2022)", secf._latest_annual(JS["facts"]["us-gaap"], secf._REV)[0] == 1200)
ok("ignores 10-Q quarterly rows", secf._latest_annual(JS["facts"]["us-gaap"], secf._REV)[1] == "2023-12-31")

print("Revenue-tag fallback:")
JS2 = {"facts": {"us-gaap": {
    "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
        {"end": "2023-12-31", "val": 500, "form": "10-K"}]}},
    "NetIncomeLoss": {"units": {"USD": [{"end": "2023-12-31", "val": -50, "form": "10-K"}]}}}}}
m2 = secf.margins_from_companyfacts(JS2)
ok("falls back to alt revenue tag", m2["npat_margin"] == round(-50/500, 4))   # -0.10
ok("fcf None when cash-flow tags absent", m2["fcf_margin"] is None)

print("Guards + gate integration:")
ok("empty payload -> None margins", secf.margins_from_companyfacts({})["npat_margin"] is None)
ok("no revenue -> None (fail closed later)",
   secf.margins_from_companyfacts({"facts": {"us-gaap": {}}})["npat_margin"] is None)
# the loss-maker above must be rejected by the live gate
ok("alt-tag loss-maker rejected by is_quality",
   cfg.is_quality(m2["npat_margin"], m2["fcf_margin"]) is False)
# the profitable TESTCO must pass
ok("profitable TESTCO passes is_quality", cfg.is_quality(m["npat_margin"], m["fcf_margin"]) is True)

print(f"\nALL {passed} SEC-FUNDAMENTALS ASSERTIONS PASSED")
