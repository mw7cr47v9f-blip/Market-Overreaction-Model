"""
Free, sustainable fundamentals for the LIVE profitability gate — sourced from the
SEC's XBRL companyfacts API (data.sec.gov), so no paid data feed is needed once the
EODHD backtest subscription is dropped. US-only (the live model is US/NYSE).

Reuses the ticker->CIK map and User-Agent header the insider module already loads.
Only two ratios are needed by cfg.is_quality: NPAT margin and FCF margin, taken from
the most recent ANNUAL (10-K) figures.

parse/margins_from_companyfacts are PURE and unit-tested offline against a synthetic
companyfacts payload; only margins_for_tickers touches the network.
"""
from __future__ import annotations

import sys
import time

from . import us_insiders

_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{:010d}.json"

# XBRL concept fallbacks (companies tag the same line differently across filers/years)
_REV = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"]
_NI = ["NetIncomeLoss", "ProfitLoss"]
_OCF = ["NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
_CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]


def log(m):
    print(f"[sec-fund] {m}", file=sys.stderr, flush=True)


def _latest_annual(gaap: dict, tags: list):
    """Most recent ANNUAL (10-K) USD value across a concept's tag fallbacks.
    Returns (value, end_date) or (None, None)."""
    for tag in tags:
        node = gaap.get(tag)
        if not isinstance(node, dict):
            continue
        best = None
        for e in (node.get("units", {}) or {}).get("USD", []) or []:
            form = str(e.get("form", ""))
            end = e.get("end")
            val = e.get("val")
            if end is None or val is None or not form.startswith("10-K"):
                continue
            if best is None or end > best[0]:
                best = (end, val)
        if best is not None:
            return best[1], best[0]
    return None, None


def margins_from_companyfacts(js) -> dict:
    """companyfacts JSON -> {npat_margin, fcf_margin} from the latest 10-K.
    SEC capex is a positive outflow, so FCF = operating cash flow - capex."""
    if not isinstance(js, dict):
        return {"npat_margin": None, "fcf_margin": None}
    gaap = ((js.get("facts", {}) or {}).get("us-gaap", {}) or {})
    rev, _ = _latest_annual(gaap, _REV)
    ni, _ = _latest_annual(gaap, _NI)
    ocf, _ = _latest_annual(gaap, _OCF)
    capex, _ = _latest_annual(gaap, _CAPEX)
    if not rev:                       # no revenue -> can't form margins (fail closed later)
        return {"npat_margin": None, "fcf_margin": None}
    npat = (ni / rev) if ni is not None else None
    fcf = ((ocf - capex) / rev) if (ocf is not None and capex is not None) else None
    return {"npat_margin": round(npat, 4) if npat is not None else None,
            "fcf_margin": round(fcf, 4) if fcf is not None else None}


def margins_for_tickers(tickers, session=None) -> dict:
    """{ticker -> {npat_margin, fcf_margin}} via SEC companyfacts. Network; live path."""
    import requests
    sess = session or requests
    us_insiders._load_cik_map()          # REQUIRED: _cik_for reads this cache, doesn't fill it
    out = {}
    for t in tickers:
        cik = us_insiders._cik_for(t)
        if not cik:
            out[t] = {"npat_margin": None, "fcf_margin": None}
            continue
        try:
            r = sess.get(_FACTS_URL.format(int(cik)), headers=us_insiders._HEADERS, timeout=30)
            out[t] = margins_from_companyfacts(r.json()) if r.status_code == 200 else \
                {"npat_margin": None, "fcf_margin": None}
        except Exception as e:  # noqa: BLE001
            log(f"{t} companyfacts failed: {e!r}")
            out[t] = {"npat_margin": None, "fcf_margin": None}
        time.sleep(0.12)              # SEC fair-access: stay under ~10 req/s
    return out
