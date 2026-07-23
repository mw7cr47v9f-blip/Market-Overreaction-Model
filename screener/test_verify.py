"""
Offline test for verify_llm: mocks the Anthropic API and the SEC filing fetch,
feeds synthetic screen outputs, and checks that the dashboard renders, the DATA
object has the shape the template expects, the email embeds the narrative, and
the fail-soft (no-API-key) path still produces a mechanical dashboard.

    python -m screener.test_verify
"""
import json
import os
import re
import tempfile

from screener import verify_llm as V
from screener import email_report as E

passed = 0


def check(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok  {name}")


CAND = {
    "scan_date": "2026-07-22",
    "count": 1,
    "candidates": [{
        "market": "US", "currency": "USD", "ticker": "ACME", "name": "Acme Corp",
        "sector": "Industrials", "market_cap": 10_100_000_000, "benchmark": "^RUT",
        "window_len": 3, "window_start": "2026-07-20", "window_end": "2026-07-22",
        "event_date": "2026-07-21", "raw_return": -0.235, "index_return": -0.01,
        "index_relative": -0.225, "z_score": -3.4, "daily_sigma": 0.02,
        "avg_daily_value": 171_000_000, "last_close": 62.45,
        "npat_margin": 0.08, "fcf_margin": 0.05,
        "director_buy": True, "director_buy_val": 250000, "favoured": True,
    }],
}
CONF = {"scan_date": "2026-07-22",
        "confirmed_today": [{"market": "US", "ticker": "ACME", "name": "Acme Corp",
                             "sector": "Industrials", "drop_date": "2026-07-21",
                             "entry_date": "2026-07-22", "entry_price": 64.10}],
        "pending": [], "expired": []}
LED = {"model": {"scan_date": "2026-07-22", "n_open": 1,
                 "open_positions": [{"market": "US", "ticker": "ACME", "name": "Acme Corp",
                                     "sector": "Industrials", "entry_date": "2026-07-22",
                                     "entry_price": 64.10, "last_price": 64.10,
                                     "ret_pct": 0.0, "days_held": 0,
                                     "sell_due_date": "2026-10-22"}],
                 "closed_this_run": [], "sell_soon": [],
                 "realised_tally": {"n_closed": 0, "win_rate": None, "avg_return": None}},
       "holdings": []}
ANN = {"market": "US", "ticker": "ACME", "cik": 123, "event_date": "2026-07-21",
       "filings": [{"date": "2026-07-21", "form": "8-K", "description": "Results of Operations",
                    "filing_url": "https://www.sec.gov/Archives/edgar/data/123/x/acme-8k.htm"}]}

FAKE_LLM = json.dumps({
    "status": "judge",
    "read": "Cap loss looks large relative to the disclosed miss; a judgement case pending the full report.",
    "trigger": "(B) Judgement required",
    "detail": "8-K disclosed a preliminary Q2 revenue miss of ~$120m and a CFO transition.",
    "implied": "~$2.4bn wiped vs a mechanical impact in the low hundreds of millions.",
    "going_concern": {"text": "none noted", "ok": 1},
    "halt": {"text": "no", "ok": 1},
})


def _seed(d):
    os.makedirs(os.path.join(d, "announcements"), exist_ok=True)
    json.dump(CAND, open(os.path.join(d, "candidates_new.json"), "w"))
    json.dump(CONF, open(os.path.join(d, "confirmations.json"), "w"))
    json.dump(LED, open(os.path.join(d, "ledger_status.json"), "w"))
    json.dump(ANN, open(os.path.join(d, "announcements", "US_ACME_2026-07-21.json"), "w"))


def _template():
    here = os.path.dirname(__file__)
    return os.path.join(here, "..", "dashboard_template.html")


def _extract_data(html):
    m = re.search(r"const DATA = (\{.*\})\s*;\s*/\* DATA_END", html, re.S)
    assert m, "DATA block not found in rendered dashboard"
    return json.loads(m.group(1))


def main():
    # ---- 1. LLM path (mocked API + filing fetch) --------------------------
    V._anthropic = lambda api_key, model, payload, max_tokens=900: FAKE_LLM
    V._fetch_filing_text = lambda url, limit=18000: "Acme preliminary Q2: revenue light; CFO transition."
    with tempfile.TemporaryDirectory() as d:
        _seed(d)
        v = V.build(d, template=_template(), model="claude-sonnet-5", api_key="test-key")
        check("llm_used True when key present", v["llm_used"] is True)
        check("model recorded", v["model"] == "claude-sonnet-5")
        check("dashboard_file set", v["dashboard_file"] == "daily_dashboard.html")
        check("ACME narrative captured", "ACME" in v["by_ticker"] and v["by_ticker"]["ACME"]["read"])
        check("trigger classified", v["by_ticker"]["ACME"]["trigger"].startswith("("))

        dash = open(os.path.join(d, "daily_dashboard.html")).read()
        check("dashboard has const DATA", "const DATA" in dash)
        DATA = _extract_data(dash)
        check("scan_date humanised", DATA["scan_date"] == "22 July 2026")
        check("one candidate card", len(DATA["candidates"]) == 1)
        c = DATA["candidates"][0]
        check("card ticker/market", c["tk"] == "ACME" and c["mk"] == "US")
        check("card drop formatted", c["drop"] == "-23.5%")
        check("card idx in pp", c["idx"].endswith("pp"))
        check("card cap humanised", c["cap"].endswith("bn"))
        check("card liq per day", c["liq"].endswith("/day"))
        check("gate has 10 rows (full stack)", len(c["gate"]) == 10)
        check("gate rows are [label,val,ok]", all(len(g) == 3 for g in c["gate"]))
        labels = [g[0] for g in c["gate"]]
        check("gate includes drop threshold", any("Drop" in l for l in labels))
        check("gate includes sector", any("sector" in l.lower() for l in labels))
        check("gate includes director ≥$50k", any("Director" in l for l in labels))
        check("quality rows pass (margins ok)",
              all(g[2] == 1 for g in c["gate"] if "NPAT" in g[0] or "cash flow" in g[0].lower()))
        check("director gate shows $250k", any("250" in str(g[1]) for g in c["gate"] if "Director" in g[0]))
        check("insider mentions $250k", "250" in c["insider"])
        check("sources carry EDGAR url", c["sources"][0][1].startswith("http"))
        check("confirmed_today mapped", DATA["confirmed_today"][0]["tk"] == "ACME")
        check("ledger open mapped", DATA["ledger"]["open"][0]["tk"] == "ACME")
        check("tally win_rate is string", isinstance(DATA["ledger"]["tally"]["win_rate"], str))

        # email embeds the narrative + attaches the dashboard
        subj, text, html, atts = E.build_email(d)
        check("email subject has session date", "22 July 2026" in subj)
        check("email text embeds read", "judgement case pending" in text)
        check("email html embeds trigger", "Judgement required" in html)
        check("email notes model verification", "SEC EDGAR" in text and "Claude" in text)
        check("dashboard attached once", len(atts) == 1)
        check("attachment named by date", atts[0]["filename"].startswith("22 July 2026"))
        check("attachment is base64 content", len(atts[0]["content"]) > 100)

    # ---- 2. fail-soft path (no API key) -----------------------------------
    with tempfile.TemporaryDirectory() as d:
        _seed(d)
        v = V.build(d, template=_template(), api_key=None)
        check("llm_used False without key", v["llm_used"] is False)
        check("model None without key", v["model"] is None)
        check("mechanical dashboard still built", os.path.exists(os.path.join(d, "daily_dashboard.html")))
        DATA = _extract_data(open(os.path.join(d, "daily_dashboard.html")).read())
        check("mechanical card still present", len(DATA["candidates"]) == 1)
        check("mechanical read mentions gates", "quality and director-buy gates" in DATA["candidates"][0]["read"])
        subj, text, html, atts = E.build_email(d)
        check("email still sends mechanically", "mechanical screen only" in text)
        check("dashboard still attached in fail-soft", len(atts) == 1)

    # ---- 3. empty day (no candidates) -------------------------------------
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "announcements"), exist_ok=True)
        json.dump({"scan_date": "2026-07-22", "count": 0, "candidates": []},
                  open(os.path.join(d, "candidates_new.json"), "w"))
        json.dump({"scan_date": "2026-07-22", "confirmed_today": [], "pending": [], "expired": []},
                  open(os.path.join(d, "confirmations.json"), "w"))
        json.dump(LED, open(os.path.join(d, "ledger_status.json"), "w"))
        v = V.build(d, template=_template(), api_key=None)
        check("empty day builds dashboard", os.path.exists(os.path.join(d, "daily_dashboard.html")))
        DATA = _extract_data(open(os.path.join(d, "daily_dashboard.html")).read())
        check("empty day zero candidates", len(DATA["candidates"]) == 0)
        check("empty day extreme dash", DATA["summary"]["extreme"]["v"] == "—")

    print(f"\nALL {passed} VERIFY ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
