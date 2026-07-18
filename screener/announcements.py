"""
Best-effort ASX announcement capture for each new candidate.

Runs on the GitHub Actions runner (full internet). For every new candidate it
pulls the company's recent announcements from the ASX platform around the event
date and writes them to data/announcements/<TICKER>_<eventdate>.json, including
the price-sensitive flag and the PDF URL. The daily Claude task then reads these
as the STARTING POINT for the primary-source analysis (it still opens the actual
announcement / the company's own results release before making any claim).

If the endpoint shape changes, this fails soft — the screen still produces
candidates; only the pre-fetched announcement convenience list is missing.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta

ASX_ANN_URL = ("https://asx.api.markitdigital.com/asx-research/1.0/companies/"
               "{code}/announcements?pageSize=30&access_token="
               "83ff96335c2d45a094df02a206a39ff4")


def log(msg: str):
    print(f"[announce] {msg}", file=sys.stderr, flush=True)


def fetch_for_candidates(candidates, data_dir: str, window_days: int = 7):
    import requests
    out_dir = os.path.join(data_dir, "announcements")
    os.makedirs(out_dir, exist_ok=True)

    for c in candidates:
        try:
            ev = date.fromisoformat(c.event_date)
            lo, hi = ev - timedelta(days=window_days), ev + timedelta(days=2)
            url = ASX_ANN_URL.format(code=c.ticker)
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            payload = r.json()
            items = payload.get("data", {}).get("items", payload.get("data", []))
            near = []
            for a in items:
                ad = _parse_date(a.get("documentDate") or a.get("date") or "")
                if ad and lo <= ad <= hi:
                    near.append({
                        "date": ad.isoformat(),
                        "headline": a.get("header") or a.get("title"),
                        "price_sensitive": bool(a.get("isPriceSensitive") or a.get("marketSensitive")),
                        "pdf_url": a.get("url") or a.get("pdfUrl"),
                        "pages": a.get("pageCount") or a.get("numberOfPages"),
                    })
            rec = {
                "ticker": c.ticker, "name": c.name, "event_date": c.event_date,
                "window": [c.window_start, c.window_end],
                "raw_return": c.raw_return, "index_relative": c.index_relative,
                "z_score": c.z_score, "announcements": near,
            }
            path = os.path.join(out_dir, f"{c.ticker}_{c.event_date}.json")
            with open(path, "w") as f:
                json.dump(rec, f, indent=2)
            log(f"{c.ticker}: {len(near)} announcement(s) near {c.event_date}")
        except Exception as e:  # noqa: BLE001
            log(f"{c.ticker}: announcement fetch failed: {e!r}")


def _parse_date(s: str):
    s = str(s)[:10]
    try:
        return date.fromisoformat(s)
    except Exception:  # noqa: BLE001
        return None
