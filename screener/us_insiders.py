"""
US insider (SEC Form 4) signal for the backtest — director / officer open-market
buying and selling in the ~6 months BEFORE an oversold event.

Question we are testing: among oversold US names, does prior insider *buying*
(directors putting their own money in on the way down) foreshadow better forward
returns, and does prior insider *selling* foreshadow worse ones? If yes, the live
model tilts (or filters) toward director-buy names.

DATA SOURCE — bulk, not per-filing. Earlier this pulled every Form-4 document one
at a time (~120k HTTP requests) and blew the CI time budget. SEC publishes the
identical data as ~one zip PER QUARTER (the "Insider Transactions Data Sets"),
each a set of tab-delimited tables. We download only the quarters spanning the
events, read three tables (SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS), filter to
our issuer CIKs, and classify — a couple dozen downloads instead of 120k, minutes
instead of hours, and MORE complete (every insider, not a capped subset).

Point-in-time integrity: Form 4 must be filed within 2 business days of the trade,
so a filing dated on/before the event genuinely predates it. We window on the
TRANSACTION date and additionally require the filing to have been public by the
event (FILING_DATE <= event), so nothing filed late leaks in.

The pure functions (parse_form4_xml, parsed_from_bulk, classify_window, net_signal,
_norm_date, quarters_between, _dataset_urls) are unit-tested offline; only
insider_signals_for_events / _download_quarter touch the network.
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import date, timedelta

_UA = os.environ.get("SEC_USER_AGENT",
                     "Market-Overreaction-Screener research contact@example.com")
_HEADERS = {"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
# Two hosting paths have been used over time; we try both per quarter.
_DATASET_HOSTS = [
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{stem}",
    "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{stem}",
]

LOOKBACK_DAYS = 182          # ~6 months of insider activity before the event
BUY_CODES = {"P"}            # open-market purchase (discretionary, conviction)
SELL_CODES = {"S"}           # open-market sale
# A (award/grant), M (option exercise), F (tax withholding), G (gift) etc. are
# NOT discretionary open-market signals, so they are excluded from the signal.

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

_cik_cache: dict[str, int] = {}


def log(m):
    print(f"[insider] {m}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Date / number normalisation (bulk TSVs use DD-MON-YYYY; XML uses ISO)
# --------------------------------------------------------------------------

def _norm_date(s):
    """Normalise a date to ISO 'YYYY-MM-DD', accepting ISO, 'DD-MON-YYYY' and
    'YYYYMMDD'. Returns None on anything unparseable."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        pass
    m = re.match(r"^(\d{1,2})-([A-Za-z]{3})-(\d{4})", s)
    if m:
        mon = _MONTHS.get(m.group(2).upper())
        if mon:
            return f"{int(m.group(3)):04d}-{mon:02d}-{int(m.group(1)):02d}"
    if re.match(r"^\d{8}$", s):
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return None


def _num(x):
    if x is None:
        return None
    try:
        v = float(str(x).strip())
        return v if v == v else None            # drop NaN
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Pure parsing / classification (unit-tested offline)
# --------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find(el, name):
    for c in el.iter():
        if _strip_ns(c.tag) == name:
            return c
    return None


def _text(el, name):
    node = _find(el, name)
    return node.text.strip() if node is not None and node.text else None


def _xval(el, name):
    node = _find(el, name)
    if node is None:
        return None
    v = _find(node, "value")
    txt = (v.text if v is not None and v.text else node.text)
    return str(txt).strip() if txt is not None else None


def parse_form4_xml(xml_str: str) -> dict:
    """Parse one Form-4 ownership XML into role flags + non-derivative
    transactions. Kept as the reference decoder (and a fallback); the bulk path
    uses parsed_from_bulk to produce the identical shape."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return {"is_director": False, "is_officer": False, "is_tenpct": False,
                "transactions": []}

    def flag(name):
        t = _text(root, name)
        return str(t).strip() in ("1", "true", "True") if t is not None else False

    role = {"is_director": flag("isDirector"), "is_officer": flag("isOfficer"),
            "is_tenpct": flag("isTenPercentOwner")}
    txns = []
    for node in root.iter():
        if _strip_ns(node.tag) != "nonDerivativeTransaction":
            continue
        code = _text(node, "transactionCode")
        if not code:
            continue
        txns.append(_mk_txn(_xval(node, "transactionDate"), code,
                            _num(_xval(node, "transactionShares")),
                            _num(_xval(node, "transactionPricePerShare")),
                            _xval(node, "transactionAcquiredDisposedCode")))
    return {**role, "transactions": txns}


def _mk_txn(d, code, shares, price, acq):
    value = (shares * price) if (shares is not None and price is not None) else None
    return {"date": _norm_date(d), "code": str(code).strip().upper(),
            "acq_disp": (str(acq).strip().upper() if acq else None),
            "shares": shares, "price": price, "value": value}


def parsed_from_bulk(relationship_values, trans_records) -> dict:
    """Build the same {is_director,is_officer,is_tenpct,transactions} shape from
    bulk-table rows. `relationship_values` = the REPORTINGOWNER relationship
    strings for the accession; `trans_records` = NONDERIV_TRANS rows as dicts with
    keys date/code/shares/price/acq."""
    j = " ".join(str(v).upper() for v in relationship_values if v is not None)
    role = {"is_director": "DIRECTOR" in j, "is_officer": "OFFICER" in j,
            "is_tenpct": "TENPERCENT" in j or "10" in j and "OWNER" in j}
    txns = [_mk_txn(t.get("date"), t.get("code", ""), _num(t.get("shares")),
                    _num(t.get("price")), t.get("acq")) for t in trans_records]
    return {**role, "transactions": txns}


def classify_window(filings: list[dict], event_date: str, lookback_days: int = LOOKBACK_DAYS) -> dict:
    """Aggregate parsed filings into an insider signal for one event.
    `filings` = [{"filing_date","parsed"}]. Counts only BUY_CODES / SELL_CODES
    transactions whose TRANSACTION date is within (event-lookback, event] AND whose
    filing was public by the event. Director/officer buys tracked separately."""
    ev = date.fromisoformat(event_date[:10])
    lo = ev - timedelta(days=lookback_days)
    buy_n = sell_n = 0
    buy_val = sell_val = 0.0
    dir_buy_n = dir_sell_n = 0
    for f in filings:
        fd = _to_date(f.get("filing_date"))
        if fd is not None and fd > ev:          # not yet public at event time
            continue
        parsed = f.get("parsed") or {}
        is_dir = parsed.get("is_director") or parsed.get("is_officer")
        for tx in parsed.get("transactions", []):
            td = _to_date(tx.get("date"))
            if td is None or not (lo < td <= ev):
                continue
            code = tx.get("code")
            val = tx.get("value") or 0.0
            if code in BUY_CODES:
                buy_n += 1
                buy_val += val
                if is_dir:
                    dir_buy_n += 1
            elif code in SELL_CODES:
                sell_n += 1
                sell_val += val
                if is_dir:
                    dir_sell_n += 1
    net_val = buy_val - sell_val
    return {
        "insider_buy_n": buy_n, "insider_sell_n": sell_n,
        "insider_buy_val": round(buy_val, 2), "insider_sell_val": round(sell_val, 2),
        "insider_net_val": round(net_val, 2),
        "director_buy": dir_buy_n > 0, "director_sell": dir_sell_n > 0,
        "insider_signal": net_signal(buy_n, sell_n, net_val),
    }


def net_signal(buy_n: int, sell_n: int, net_val: float) -> str:
    if buy_n == 0 and sell_n == 0:
        return "none"
    if buy_n > 0 and net_val > 0:
        return "buy"
    if sell_n > 0 and net_val < 0:
        return "sell"
    return "mixed"


def _to_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return _to_date(_norm_date(s)) if _norm_date(s) else None


# --------------------------------------------------------------------------
# Quarter maths / dataset URLs (pure)
# --------------------------------------------------------------------------

def quarters_between(lo: date, hi: date):
    """Inclusive list of (year, quarter) covering [lo, hi]."""
    y, q = lo.year, (lo.month - 1) // 3 + 1
    out = []
    while (y, q) <= (hi.year, (hi.month - 1) // 3 + 1):
        out.append((y, q))
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out


def _dataset_urls(y: int, q: int):
    stem = f"{y}q{q}_form345.zip"
    return [h.format(stem=stem) for h in _DATASET_HOSTS]


# --------------------------------------------------------------------------
# Network layer (GitHub Actions only)
# --------------------------------------------------------------------------

def _load_cik_map():
    import requests
    if _cik_cache:
        return
    r = requests.get(_TICKER_MAP_URL, timeout=30, headers=_HEADERS)
    r.raise_for_status()
    for row in r.json().values():
        _cik_cache[str(row["ticker"]).upper()] = int(row["cik_str"])
    log(f"loaded {len(_cik_cache)} ticker->CIK mappings")


def _cik_for(ticker: str):
    """Ticker -> issuer CIK, tolerant of '.'/'-' class-share punctuation."""
    t = ticker.upper().strip()
    for cand in (t, t.replace(".", "-"), t.replace("-", "."),
                 t.replace(".", ""), t.replace("-", "")):
        if cand in _cik_cache:
            return _cik_cache[cand]
    return None


def _read_table(z: zipfile.ZipFile, prefix: str, want_cols: set, dtype=str):
    import pandas as pd
    fn = next((n for n in z.namelist() if os.path.basename(n).upper().startswith(prefix)), None)
    if fn is None:
        return None
    with z.open(fn) as fh:
        return pd.read_csv(fh, sep="\t", dtype=dtype, on_bad_lines="skip",
                           usecols=lambda c: c.strip().upper() in want_cols)


def _download_quarter(y: int, q: int, session):
    for url in _dataset_urls(y, q):
        try:
            r = session.get(url, timeout=180, headers=_HEADERS)
            if r.status_code == 200 and r.content:
                return zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:  # noqa: BLE001
            log(f"  {y}Q{q} {url[-40:]} failed: {e!r}")
    log(f"  {y}Q{q}: no dataset reachable")
    return None


def _index_quarter(z, want_ciks: dict) -> dict:
    """Return {ticker: [{filing_date, parsed}]} for accessions whose issuer CIK is
    in want_ciks (a {cik:int -> event_ticker} map)."""
    import pandas as pd
    out = defaultdict(list)
    sub = _read_table(z, "SUBMISSION",
                      {"ACCESSION_NUMBER", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "FILING_DATE"})
    if sub is None or sub.empty:
        return out
    sub.columns = [c.strip().upper() for c in sub.columns]
    sub["_cik"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
    sub = sub[sub["_cik"].isin(want_ciks.keys())]
    if sub.empty:
        return out
    accs = set(sub["ACCESSION_NUMBER"])
    acc_meta = {r["ACCESSION_NUMBER"]: (want_ciks.get(int(r["_cik"])), _norm_date(r["FILING_DATE"]))
                for _, r in sub.iterrows()}

    own = _read_table(z, "REPORTINGOWNER", {"ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP"})
    rel_by_acc = defaultdict(list)
    if own is not None and not own.empty:
        own.columns = [c.strip().upper() for c in own.columns]
        own = own[own["ACCESSION_NUMBER"].isin(accs)]
        for _, r in own.iterrows():
            rel_by_acc[r["ACCESSION_NUMBER"]].append(r.get("RPTOWNER_RELATIONSHIP"))

    tr = _read_table(z, "NONDERIV_TRANS",
                     {"ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                      "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"})
    trans_by_acc = defaultdict(list)
    if tr is not None and not tr.empty:
        tr.columns = [c.strip().upper() for c in tr.columns]
        tr = tr[tr["ACCESSION_NUMBER"].isin(accs)]
        for _, r in tr.iterrows():
            trans_by_acc[r["ACCESSION_NUMBER"]].append(
                {"date": r.get("TRANS_DATE"), "code": r.get("TRANS_CODE"),
                 "shares": r.get("TRANS_SHARES"), "price": r.get("TRANS_PRICEPERSHARE"),
                 "acq": r.get("TRANS_ACQUIRED_DISP_CD")})

    for acc, (ticker, fdate) in acc_meta.items():
        if ticker is None:
            continue
        parsed = parsed_from_bulk(rel_by_acc.get(acc, []), trans_by_acc.get(acc, []))
        out[ticker].append({"filing_date": fdate, "parsed": parsed})
    return out


def insider_signals_for_events(events_by_ticker: dict, years: float = 5) -> dict:
    """events_by_ticker: {TICKER: [event_date_iso, ...]}. Returns
    {(TICKER, event_date): classify_window(...)}. US only, from SEC bulk datasets.
    Best-effort: unreachable quarters simply contribute no filings."""
    import requests
    out = {}
    if not events_by_ticker:
        return out
    try:
        _load_cik_map()
    except Exception as e:  # noqa: BLE001
        log(f"CIK map load failed, skipping insider signal: {e!r}")
        return out

    want_ciks = {}
    for tk in events_by_ticker:
        cik = _cik_for(tk)
        if cik is not None:
            want_ciks[cik] = tk
    log(f"insider: {len(want_ciks)}/{len(events_by_ticker)} tickers mapped to CIK")
    if not want_ciks:
        return out

    all_dates = [d for evs in events_by_ticker.values() for e in evs
                 if (d := _to_date(e)) is not None]
    if not all_dates:
        return out
    lo = min(all_dates) - timedelta(days=LOOKBACK_DAYS + 10)
    hi = max(all_dates)
    qs = quarters_between(lo, hi)
    log(f"insider: scanning {len(qs)} quarters {qs[0]}..{qs[-1]}")

    session = requests.Session()
    filings_by_ticker = defaultdict(list)
    for (y, q) in qs:
        t0 = time.time()
        z = _download_quarter(y, q, session)
        if z is None:
            continue
        idx = _index_quarter(z, want_ciks)
        for tk, lst in idx.items():
            filings_by_ticker[tk].extend(lst)
        log(f"insider: {y}Q{q} -> {sum(len(v) for v in idx.values())} filings "
            f"for {len(idx)} names ({time.time()-t0:.0f}s)")

    for tk, evs in events_by_ticker.items():
        filings = filings_by_ticker.get(tk, [])
        for ev in evs:
            out[(tk, ev)] = classify_window(filings, ev)
    log(f"insider signals for {len(out)} (ticker,event) pairs from "
        f"{sum(len(v) for v in filings_by_ticker.values())} filings")
    return out
