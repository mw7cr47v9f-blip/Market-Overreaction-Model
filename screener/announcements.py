"""
Best-effort ASX announcement capture for each new candidate.

Runs on the GitHub Actions runner (full internet). For every new candidate it
pulls the company's recent announcements from the ASX platform around the event
date and writes them to data/announcements/<TICKER>_<eventdate>.json, including
the headline, price-sensitive flag, and a direct PDF link. The daily Claude task
then reads these as the STARTING POINT for the primary-source analysis (it still
opens the actual announcement before making any claim).

Field names confirmed against the live API (July 2026): each item carries
`headline`, `date` (ISO, UTC), `isPriceSensitive`, `documentKey`, `fileSize`.
The item's own `url` field is empty, so the PDF link is built from
`documentKey` via the file endpoint (verified to resolve to the real PDF).

If the endpoint shape changes, this fails soft — the screen still produces
candidates; only the pre-fetched announcement convenience list is affected.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta

_TOKEN = "83ff96335c2d45a094df02a206a39ff4"
ASX_ANN_URL = ("https://asx.api.markitdigital.com/asx-research/1.0/companies/"
               "{code}/announcements?pageSize=30&access_token=" + _TOKEN)
# documentKey -> primary PDF (verified: returns the actual lodged document)
ASX_FILE_URL = ("https://asx.api.markitdigital.com/asx-research/1.0/file/"
                "{key}?access_token=" + _TOKEN)


def log(msg: str):
    print(f"[announce] {msg}", file=sys.stderr, flush=True)


def fetch_for_candidates(candidates, data_dir: str, window_days: int = 7):
    import requests
    out_dir = os.path.join(data_dir, "announcements")
    os.makedirs(out_dir, exist_ok=True)

    for c in candidates:
        try:
            ev = date.fromisoformat(c.event_date)
            lo, hi = ev - timedelta(days=window_days), ev + timedelta(days=3)
            url = ASX_ANN_URL.format(code=c.ticker)
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            payload = r.json()
            items = _items(payload)
            near = []
            for a in items:
                ad = _parse_date(a.get("date"))
                if ad and lo <= ad <= hi:
                    key = a.get("documentKey")
                    near.append({
                        "date": (a.get("date") or "")[:10],
                        "datetime_utc": a.get("date"),
                        "headline": a.get("headline"),
                        "type": a.get("announcementType"),
                        "price_sensitive": bool(a.get("isPriceSensitive")),
                        "document_key": key,
                        "pdf_url": ASX_FILE_URL.format(key=key) if key else None,
                        "file_size": a.get("fileSize"),
                    })
            rec = {
                "ticker": c.ticker, "name": c.name, "event_date": c.event_date,
                "window": [c.window_start, c.window_end],
                "raw_return": c.raw_return, "index_relative": c.index_relative,
                "z_score": c.z_score, "announcements": near,
            }
            path = os.path.join(out_dir, f"{getattr(c, 'market', 'ASX')}_{c.ticker}_{c.event_date}.json")
            with open(path, "w") as f:
                json.dump(rec, f, indent=2)
            ps = sum(1 for a in near if a["price_sensitive"])
            log(f"{c.ticker}: {len(near)} announcement(s) near {c.event_date} ({ps} price-sensitive)")
        except Exception as e:  # noqa: BLE001
            log(f"{c.ticker}: announcement fetch failed: {e!r}")


def _items(payload):
    """The API nests items under data.items; tolerate a couple of shapes."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", payload)
    if isinstance(data, dict):
        return data.get("items", data.get("announcements", []))
    if isinstance(data, list):
        return data
    return []


def _parse_date(s):
    """`date` is an ISO UTC timestamp, e.g. 2026-07-15T22:38:13.000Z."""
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:  # noqa: BLE001
        return None
