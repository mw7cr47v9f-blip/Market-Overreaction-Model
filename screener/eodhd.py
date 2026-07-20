"""
EODHD fundamentals — the reliable, point-in-time replacement for the yfinance
fundamentals fetch (which Yahoo now rate-limits into uselessness).

Only the fetch functions touch the network (they run on GitHub Actions with the
EODHD_API_TOKEN secret). parse_fundamentals / _fy_records are PURE and unit-tested
offline against a synthetic EODHD payload.

Point-in-time: EODHD stamps each statement with a `filing_date` (when it was
actually reported). metrics_for_event uses that, so a statement only becomes
"known" after it was filed — no lookahead.
"""
from __future__ import annotations

import os
import sys

_BASE = "https://eodhd.com/api"
# screener market -> EODHD exchange suffix
_EXCHANGE = {"US": "US", "ASX": "AU", "TSX": "TO", "LSE": "LSE", "FTSE": "LSE"}

# OTC / pink-sheet markers in EODHD's per-symbol "Exchange" field. Names on these
# venues are illiquid by definition and would fail the model's liquidity floor
# anyway, so excluding them costs NO real coverage (no survivorship bias — they
# were never tradeable candidates) while capping the price-fetch count. Matched
# case-insensitively as substrings so venue-string variants (OTCQB, OTCMKTS,
# PINK, GREY, OTC GREY, EXPM, NMFQS ...) are all caught.
_OTC_MARKERS = ("OTC", "PINK", "GREY", "NMFQS", "EXPM")


def _venue_family(exchange) -> str:
    """Collapse EODHD's per-symbol venue string into a canonical family so we can
    scope the universe cleanly. Order matters: the specific NYSE variants (MKT =
    AMEX, ARCA) MUST be tested before the bare 'NYSE' catch, or they'd fold into
    NYSE. Unknown strings are returned upper-cased so nothing is silently dropped
    without showing up in the composition log."""
    e = str(exchange or "").upper().strip()
    if not e:
        return "(none)"
    if any(m in e for m in _OTC_MARKERS):
        return "OTC"
    if "ARCA" in e:
        return "ARCA"
    if "MKT" in e or "AMEX" in e or "AMERICAN" in e:   # NYSE MKT / NYSE American = AMEX
        return "AMEX"
    if "NASDAQ" in e or "NMS" in e or "NGS" in e or "NCM" in e:
        return "NASDAQ"
    if "NYSE" in e or "NEW YORK" in e:
        return "NYSE"
    return e


def log(m):
    print(f"[eodhd] {m}", file=sys.stderr, flush=True)


def token():
    return os.environ.get("EODHD_API_TOKEN")


def _f(v):
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Pure parsing (offline-tested)
# --------------------------------------------------------------------------

def _shares_by_year(js) -> dict:
    """{'2024': shares} from outstandingShares.annual, best-effort."""
    out = {}
    ann = (js.get("outstandingShares", {}) or {}).get("annual", {}) or {}
    for _, row in (ann.items() if isinstance(ann, dict) else enumerate(ann)):
        if not isinstance(row, dict):
            continue
        d = str(row.get("dateFormatted") or row.get("date") or "")[:4]
        sh = _f(row.get("shares")) or (_f(row.get("sharesMln")) * 1e6 if _f(row.get("sharesMln")) else None)
        if d and sh:
            out[d] = sh
    return out


def _fy_records(js) -> list:
    """Merge yearly income / balance / cash-flow into one ordered list of FY dicts."""
    fin = js.get("Financials", {}) or {}
    inc = (fin.get("Income_Statement", {}) or {}).get("yearly", {}) or {}
    bal = (fin.get("Balance_Sheet", {}) or {}).get("yearly", {}) or {}
    cfs = (fin.get("Cash_Flow", {}) or {}).get("yearly", {}) or {}
    shares_yr = _shares_by_year(js)
    recs = []
    for d, ii in inc.items():
        ii = ii or {}
        bb = bal.get(d, {}) or {}
        cc = cfs.get(d, {}) or {}
        ni = _f(ii.get("netIncome"))
        rev = _f(ii.get("totalRevenue"))
        eq = _f(bb.get("totalStockholderEquity"))
        debt = _f(bb.get("totalDebt"))
        if debt is None:
            debt = (_f(bb.get("longTermDebt")) or 0) + (_f(bb.get("shortLongTermDebt")) or 0)
        fcf = _f(cc.get("freeCashFlow"))
        if fcf is None:
            ocf = _f(cc.get("totalCashFromOperatingActivities"))
            capex = _f(cc.get("capitalExpenditures"))
            fcf = (ocf + capex) if (ocf is not None and capex is not None) else None  # capex negative
        sh = shares_yr.get(str(d)[:4])
        eps = (ni / sh) if (ni is not None and sh) else None
        recs.append({"date": str(d), "filing_date": str(ii.get("filing_date") or bb.get("filing_date") or d),
                     "ni": ni, "rev": rev, "eq": eq, "debt": debt, "fcf": fcf, "eps": eps})
    recs = [r for r in recs if r["date"]]
    recs.sort(key=lambda r: r["date"])
    return recs


def parse_fundamentals(js) -> dict:
    """EODHD fundamentals JSON -> the fund dict metrics_for_event expects, with a
    point-in-time filing_date per fiscal year."""
    if not isinstance(js, dict) or not js:
        return None
    gen = js.get("General", {}) or {}
    sector = gen.get("Sector") or gen.get("GicSector")
    shares_out = _f((js.get("SharesStats", {}) or {}).get("SharesOutstanding"))
    fy = _fy_records(js)
    if not fy:
        return None
    return {"fy": fy, "shares_out": shares_out, "sector": sector, "dividends": None}


def parse_dividends(js) -> list:
    """EODHD /div payload (list of {date, value}) -> [(iso_date, amount)]."""
    out = []
    if isinstance(js, list):
        for row in js:
            d = str((row or {}).get("date") or "")[:10]
            v = _f((row or {}).get("value"))
            if d and v is not None:
                out.append((d, v))
    return out


# --------------------------------------------------------------------------
# Network (GitHub Actions only)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Universe (incl. delisted) + EOD prices — the survivorship-free data layer
# --------------------------------------------------------------------------

def _get(session, path, params, timeout=60):
    import requests
    sess = session or requests
    r = sess.get(f"{_BASE}/{path}", params={**params, "api_token": token(), "fmt": "json"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_symbol_list(js, common_only=True) -> list:
    """EODHD exchange-symbol-list rows -> [{code,name,type,exchange,currency}]."""
    out = []
    for row in (js or []):
        if not isinstance(row, dict):
            continue
        code = str(row.get("Code", "")).strip()
        typ = str(row.get("Type", "")).strip()
        if not code:
            continue
        if common_only and typ and typ.lower() != "common stock":
            continue
        out.append({"code": code, "name": str(row.get("Name", "")).strip(),
                    "type": typ, "exchange": row.get("Exchange"), "currency": row.get("Currency")})
    return out


def _is_otc(exchange) -> bool:
    e = str(exchange or "").upper()
    return any(m in e for m in _OTC_MARKERS)


def universe(market="US", include_delisted=True, exchanges=("NYSE",), session=None):
    """Full common-stock universe for an exchange INCLUDING delisted names — the
    key to a survivorship-free backtest. Returns a DataFrame[code,name,yahoo,exchange,family].

    exchanges: which canonical venue families to KEEP (see _venue_family). Default
    ("NYSE",) — the deliberate scope for this backtest (AMEX micro-caps, ARCA and
    OTC are all excluded; NASDAQ is shelved). Names outside these families fail the
    model's liquidity/scope intent anyway, so excluding them adds NO survivorship
    bias while bounding the price-fetch count and wall-clock. Pass None to keep all."""
    import pandas as pd
    import requests
    if not token():
        return None
    ex = _EXCHANGE.get(market, "US")
    s = session or requests.Session()
    rows = parse_symbol_list(_get(s, f"exchange-symbol-list/{ex}", {}))
    n_active = len(rows)
    if include_delisted:
        try:
            rows += parse_symbol_list(_get(s, f"exchange-symbol-list/{ex}", {"delisted": 1}))
        except Exception as e:  # noqa: BLE001
            log(f"delisted list failed: {e!r}")
    df = pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)
    df["family"] = df["exchange"].map(_venue_family) if "exchange" in df.columns else "(none)"

    # Composition breakdown by canonical family — so the probe run reveals EODHD's
    # real venue strings AND how the allow-list will land, before the full run.
    try:
        vc = df["family"].value_counts()
        log(f"venue-family composition ({len(vc)} families):")
        for name, cnt in vc.items():
            log(f"    {name!s:<10} {int(cnt)}")
    except Exception:  # noqa: BLE001
        pass

    if exchanges is not None:
        keep = {str(x).upper().strip() for x in exchanges}
        before = len(df)
        df = df[df["family"].isin(keep)].reset_index(drop=True)
        log(f"kept families {sorted(keep)}: {before} -> {len(df)} common stocks "
            f"(dropped {before - len(df)} out-of-scope — no survivorship cost)")

    df["yahoo"] = df["code"]
    df["exchange_mkt"] = market
    log(f"EODHD {ex} universe: {len(df)} common stocks (active + delisted; "
        f"families kept: {sorted({str(x).upper().strip() for x in exchanges}) if exchanges else 'ALL'})")
    return df


def parse_eod(js):
    """EODHD /eod payload -> DataFrame[Close, Volume] (Close = adjusted_close, so
    splits & dividends are handled). Returns None if empty."""
    import pandas as pd
    if not isinstance(js, list) or not js:
        return None
    idx, close, vol = [], [], []
    for row in js:
        if not isinstance(row, dict):
            continue
        d = row.get("date")
        c = row.get("adjusted_close")
        if c is None:
            c = row.get("close")
        if d is None or c is None:
            continue
        idx.append(pd.Timestamp(d))
        close.append(_f(c))
        vol.append(_f(row.get("volume")) or 0.0)
    if not idx:
        return None
    return pd.DataFrame({"Close": close, "Volume": vol},
                        index=pd.DatetimeIndex(idx)).sort_index()


def eod_series(code, market, start, end, session=None):
    ex = _EXCHANGE.get(market, "US")
    try:
        js = _get(session, f"eod/{code}.{ex}", {"from": start, "to": end, "period": "d"}, timeout=30)
        return parse_eod(js)
    except Exception:  # noqa: BLE001
        return None


def fundamentals(code, market, session=None):
    tok = token()
    if not tok:
        return None
    import requests
    ex = _EXCHANGE.get(market, "US")
    sess = session or requests
    try:
        r = sess.get(f"{_BASE}/fundamentals/{code}.{ex}",
                     params={"api_token": tok, "fmt": "json"}, timeout=30)
        if r.status_code != 200:
            return None
        fund = parse_fundamentals(r.json())
        if fund is None:
            return None
        # trailing dividends for the div-yield factor
        try:
            d = sess.get(f"{_BASE}/div/{code}.{ex}",
                         params={"api_token": tok, "fmt": "json"}, timeout=30)
            if d.status_code == 200:
                fund["dividends"] = parse_dividends(d.json())
        except Exception:  # noqa: BLE001
            pass
        return fund
    except Exception as e:  # noqa: BLE001
        log(f"{code}.{ex} fundamentals failed: {e!r}")
        return None
