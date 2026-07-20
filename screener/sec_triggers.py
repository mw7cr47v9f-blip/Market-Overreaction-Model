"""
Tag each backtest event with the KIND of bad news that triggered the drop, using
the numbered item codes on the company's SEC Form 8-K filed around the event date.

Thesis under test (Keegan): genuine overreactions — sentiment shocks where the
business is intact (earnings miss, guidance cut) — mean-revert and are worth buying;
structural repricings (lost contract, impairment, distress, or corporate/M&A action
whose price move reflects deal terms) do NOT revert and should be avoided. If true, a
trigger-type filter lifts return AND cuts the left tail, like the profitability gate.

Runs on GitHub Actions (SEC access). Pure parsing/matching functions are unit-tested
offline; only fetch_8k_rows touches the network. Per-company fetch (cached), then each
of that company's events is matched to nearby 8-Ks — so ~3.4k fetches, not 25k.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta

from . import us_insiders

_SUB_URL = "https://data.sec.gov/submissions/CIK{:010d}.json"
_SUB_FILE = "https://data.sec.gov/submissions/{}"

# 8-K item code (prefix) -> trigger category. Aligned to the recoverability thesis.
_ITEM_CAT = {
    "2.02": "earnings",       # Results of Operations & Financial Condition
    "7.01": "guidance",       # Reg FD (guidance / outlook)
    "1.02": "contract_loss",  # Termination of a Material Definitive Agreement
    "1.01": "agreement",      # Entry into a Material Agreement (ambiguous: deal/financing)
    "2.01": "corporate",      # Completion of Acquisition / Disposition
    "2.05": "cost_impair",    # Costs Associated with Exit or Disposal
    "2.06": "cost_impair",    # Material Impairments
    "1.03": "distress",       # Bankruptcy or Receivership
    "2.04": "distress",       # Triggering events accelerating a debt obligation
    "3.01": "distress",       # Delisting / failure to satisfy listing rule
    "4.01": "distress",       # Changes in Registrant's Certifying Accountant
    "4.02": "distress",       # Non-Reliance on Previously Issued Financials (restatement)
    "2.03": "financing",      # Creation of a Direct Financial Obligation
    "3.02": "financing",      # Unregistered Sales of Equity (dilution)
    "3.03": "financing",      # Material Modification to Rights of Security Holders
    "5.02": "management",     # Departure/Election of Directors or Officers
    "8.01": "other",          # Other Events (regulatory, litigation, competitor — catch-all)
}
# When several categories are present, this is the priority for the single "primary".
_PRIORITY = ["earnings", "guidance", "cost_impair", "contract_loss", "distress",
             "corporate", "financing", "management", "agreement", "other"]


def log(m):
    print(f"[sec-trig] {m}", file=sys.stderr, flush=True)


def categories_for_items(items_str) -> set:
    """'2.02,9.01' -> {'earnings'}. Unknown/uncovered codes (e.g. 9.01 exhibits) drop."""
    out = set()
    for tok in str(items_str or "").replace(" ", "").split(","):
        tok = tok.strip()
        # item codes look like '2.02'; normalise '2.02.' or 'Item2.02'
        for k in _ITEM_CAT:
            if tok == k or tok.endswith(k):
                out.add(_ITEM_CAT[k])
    return out


def primary_category(cats) -> str:
    for c in _PRIORITY:
        if c in cats:
            return c
    return "none"


def _rows_from_arrays(d) -> list:
    """A submissions parallel-array block -> [{date, form, items}] for 8-K rows only."""
    forms = d.get("form", []) or []
    dates = d.get("filingDate", []) or []
    items = d.get("items", []) or []
    out = []
    for i, f in enumerate(forms):
        if not str(f).startswith("8-K"):
            continue
        out.append({"date": dates[i] if i < len(dates) else None,
                    "form": f, "items": items[i] if i < len(items) else ""})
    return out


def parse_submissions(js):
    """SEC submissions JSON -> (list of recent 8-K rows, list of older-file names to fetch)."""
    if not isinstance(js, dict):
        return [], []
    filings = js.get("filings", {}) or {}
    rows = _rows_from_arrays(filings.get("recent", {}) or {})
    older = [f.get("name") for f in (filings.get("files", []) or []) if f.get("name")]
    return rows, older


def triggers_near(rows_8k, event_date, lo_days=-6, hi_days=2) -> dict:
    """Union of 8-K item categories filed within [event+lo, event+hi] calendar days.
    Returns {items, cats, primary}. Empty -> primary 'none' (no coded 8-K found)."""
    try:
        ev = datetime.fromisoformat(str(event_date)[:10]).date()
    except ValueError:
        return {"items": "", "cats": "", "primary": "none"}
    lo, hi = ev + timedelta(days=lo_days), ev + timedelta(days=hi_days)
    items, cats = set(), set()
    for r in rows_8k:
        try:
            d = datetime.fromisoformat(str(r["date"])[:10]).date()
        except (ValueError, TypeError, KeyError):
            continue
        if lo <= d <= hi:
            for tok in str(r.get("items", "")).replace(" ", "").split(","):
                if tok:
                    items.add(tok)
            cats |= categories_for_items(r.get("items", ""))
    return {"items": "|".join(sorted(items)), "cats": "|".join(sorted(cats)),
            "primary": primary_category(cats)}


# --------------------------------------------------------------------------
# Network (GitHub Actions only)
# --------------------------------------------------------------------------

def fetch_8k_rows(cik, session=None) -> list:
    """All 8-K {date, form, items} for a CIK, merging recent + older submission files."""
    import requests
    sess = session or requests
    try:
        r = sess.get(_SUB_URL.format(int(cik)), headers=us_insiders._HEADERS, timeout=30)
        if r.status_code != 200:
            return []
        rows, older = parse_submissions(r.json())
    except Exception as e:  # noqa: BLE001
        log(f"CIK {cik} submissions failed: {e!r}")
        return []
    for fname in older:
        try:
            r2 = sess.get(_SUB_FILE.format(fname), headers=us_insiders._HEADERS, timeout=30)
            if r2.status_code == 200:
                rows += _rows_from_arrays(r2.json())
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.12)
    return rows


def enrich(events_csv, out_csv, limit=0):
    """Read committed events, tag each with its 8-K trigger, write a tagged CSV."""
    import pandas as pd
    import requests
    df = pd.read_csv(events_csv, low_memory=False)
    if limit:
        df = df.head(limit)
    sess = requests.Session()
    tickers = [t for t in df["ticker"].dropna().unique()]
    log(f"tagging {len(df)} events across {len(tickers)} tickers")
    cache = {}
    for j, tk in enumerate(tickers):
        cik = us_insiders._cik_for(str(tk).split("_")[0].split("-")[0])  # strip _old / -WS suffixes
        cache[tk] = fetch_8k_rows(cik, sess) if cik else []
        time.sleep(0.05)
        if (j + 1) % 500 == 0:
            log(f"fetched 8-K history {j+1}/{len(tickers)}")
    prim, cats, items = [], [], []
    for _, row in df.iterrows():
        t = triggers_near(cache.get(row["ticker"], []), row["date"])
        prim.append(t["primary"]); cats.append(t["cats"]); items.append(t["items"])
    df["trigger_primary"] = prim
    df["trigger_cats"] = cats
    df["trigger_items"] = items
    df.to_csv(out_csv, index=False)
    vc = df["trigger_primary"].value_counts()
    log(f"wrote {out_csv}; trigger mix:\n{vc.to_string()}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/backtest_events.csv")
    ap.add_argument("--out", default="data/backtest_events_tagged.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        from . import test_triggers  # noqa: F401
        return
    enrich(a.events, a.out, a.limit)


if __name__ == "__main__":
    main()
