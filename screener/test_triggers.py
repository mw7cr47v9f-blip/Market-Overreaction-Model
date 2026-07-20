"""Offline tests for 8-K trigger tagging. python -m screener.test_triggers"""
from screener import sec_triggers as st

passed = 0
def ok(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print("  ok ", name)


print("Item-code -> category:")
ok("2.02 -> earnings", st.categories_for_items("2.02,9.01") == {"earnings"})
ok("1.02 -> contract_loss", st.categories_for_items("1.02") == {"contract_loss"})
ok("2.06 -> cost_impair", st.categories_for_items("2.06") == {"cost_impair"})
ok("multi items union", st.categories_for_items("2.02,5.02") == {"earnings", "management"})
ok("exhibits-only 9.01 -> empty", st.categories_for_items("9.01") == set())
ok("empty -> empty", st.categories_for_items("") == set())

print("Primary-category priority:")
ok("earnings beats management", st.primary_category({"earnings", "management"}) == "earnings")
ok("cost_impair beats corporate", st.primary_category({"corporate", "cost_impair"}) == "cost_impair")
ok("empty -> none", st.primary_category(set()) == "none")

print("Submissions parsing:")
JS = {"filings": {
    "recent": {
        "form": ["8-K", "10-Q", "8-K", "4"],
        "filingDate": ["2020-03-10", "2020-02-01", "2020-03-30", "2020-03-11"],
        "items": ["2.02,9.01", "", "1.02", ""]},
    "files": [{"name": "CIK0000000001-submissions-001.json"}]}}
rows, older = st.parse_submissions(JS)
ok("only 8-K rows kept", len(rows) == 2)
ok("older files listed", older == ["CIK0000000001-submissions-001.json"])
ok("items carried", rows[0]["items"] == "2.02,9.01")

print("Event -> nearby trigger matching:")
# event on 2020-03-12: the 2.02 (Mar 10) is inside [-6,+2]; the 1.02 (Mar 30) is not.
t = st.triggers_near(rows, "2020-03-12")
ok("earnings 8-K two days before is matched", t["primary"] == "earnings")
ok("far-away 1.02 not matched", "contract_loss" not in t["cats"])
# event on 2020-03-31: now the 1.02 (Mar 30) is the nearby one
t2 = st.triggers_near(rows, "2020-03-31")
ok("contract_loss matched when near", t2["primary"] == "contract_loss")
# no 8-K anywhere near -> none
t3 = st.triggers_near(rows, "2019-01-01")
ok("no nearby 8-K -> none", t3["primary"] == "none")

print(f"\nALL {passed} TRIGGER ASSERTIONS PASSED")
