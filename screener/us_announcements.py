"""
Best-effort SEC EDGAR filing capture for US candidates — the primary-source
starting point for Steps 2-3 on US names (the equivalent of the ASX
announcements platform for ASX names).

For each new US candidate it maps ticker -> CIK, pulls the company's recent
filings around the event date (8-K, 10-Q, 10-K, etc.) and writes them to
data/announcements/US_<TICKER>_<eventdate>.json with a direct link to the
primary document. The daily analysis task reads these, then opens the actual
filing on EDGAR before making any claim.

SEC asks for a descriptive User-Agent with contact info; set SEC_USER_AGENT in
the workflow env to your own, or the default below is used.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta

_UA = os.environ.get("SEC_USER_AGENT",
                     "Market-Overreaction-Screener research contact@example.com")
_HEADERS = {"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

# filings worth surfacing for an overreaction (price-sensitive events)
_INTERESTING = {"8-K", "10-Q", "10-K", "6-K", "8-K/A", "10-Q/A"}

_cik_cache: dict[str, int] = {}


def log(msg: str):
    print(f"[sec] {msg}", file=sys.stderr, flush=True)


def _load_cik_map():
    import requests
    if _cik_cache:
        return
    r = requests.get(_TICKER_MAP_URL, timeout=30, headers=_HEADERS)
    r.raise_for_status()
    for row in r.json().values():
        _cik_cache[str(row["ticker"]).upper()] = int(row["cik_str"])
    log(f"loaded {len(_cik_cache)} ticker->CIK mappings")


def fetch_for_candidates(candidates, data_dir: str, window_days: int = 7):
    import requests
    out_dir = os.path.join(data_dir, "announcements")
    os.makedirs(out_dir, exist_ok=True)
    try:
        _load_cik_map()
    except Exception as e:  # noqa: BLE001
        log(f"could not load CIK map: {e!r}")
        return

    for c in candidates:
        try:
            cik = _cik_cache.get(c.ticker.upper())
            if cik is None:
                log(f"{c.ticker}: no CIK found")
                continue
            ev = date.fromisoformat(c.event_date)
            lo, hi = ev - timedelta(days=window_days), ev + timedelta(days=3)
            sub = requests.get(_SUBMISSIONS_URL.format(cik=cik), timeout=30, headers=_HEADERS).json()
            rec = sub.get("filings", {}).get("recent", {})
            forms = rec.get("form", [])
            dates = rec.get("filingDate", [])
            accs = rec.get("accessionNumber", [])
            docs = rec.get("primaryDocument", [])
            descs = rec.get("primaryDocDescription", [])
            near = []
            for i, form in enumerate(forms):
                fd = _parse_date(dates[i] if i < len(dates) else None)
                if not fd or not (lo <= fd <= hi):
                    continue
                if form not in _INTERESTING:
                    continue
                acc = accs[i] if i < len(accs) else ""
                doc = docs[i] if i < len(docs) else ""
                near.append({
                    "date": fd.isoformat(),
                    "form": form,
                    "description": descs[i] if i < len(descs) else None,
                    "filing_url": _ARCHIVE.format(cik=cik, acc_nodash=acc.replace("-", ""), doc=doc)
                                  if acc and doc else None,
                })
            record = {
                "market": "US", "ticker": c.ticker, "name": c.name, "cik": cik,
                "event_date": c.event_date, "window": [c.window_start, c.window_end],
                "raw_return": c.raw_return, "index_relative": c.index_relative,
                "z_score": c.z_score, "filings": near,
            }
            path = os.path.join(out_dir, f"US_{c.ticker}_{c.event_date}.json")
            with open(path, "w") as f:
                json.dump(record, f, indent=2)
            log(f"{c.ticker}: {len(near)} filing(s) near {c.event_date}")
        except Exception as e:  # noqa: BLE001
            log(f"{c.ticker}: EDGAR fetch failed: {e!r}")


def _parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:  # noqa: BLE001
        return None
