"""
US insider (SEC Form 4) signal for the backtest — director / officer open-market
buying and selling in the ~6 months BEFORE an oversold event.

Question we are testing: among oversold US names, does prior insider *buying*
(directors putting their own money in on the way down) foreshadow better forward
returns, and does prior insider *selling* foreshadow worse ones? If yes, the live
model doubles the position when a director has been buying.

Design for a backtest that runs over thousands of events under SEC's 10 req/s
rate limit:
  * Fetch each event-ticker's Form-4 list ONCE (submissions API), then fetch each
    Form-4 primary XML ONCE, parse it into dated transactions, and cache.
  * Per event, just sum the cached transactions that fall in the prior window —
    no extra network. So requests scale with (# Form-4 filings on event-tickers),
    not with (# events).

Point-in-time integrity: Form 4 must be filed within 2 business days of the
trade, so a filing dated on/before the event genuinely predates it — no lookahead.
We window on the TRANSACTION date and additionally require the filing to have
been public by the event (filingDate <= event), so nothing filed late leaks in.

Parsing (parse_form4_xml / classify_window / net_signal) is pure and unit-tested
offline; only insider_signals_for_events touches the network.
"""
from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta

_UA = os.environ.get("SEC_USER_AGENT",
                     "Market-Overreaction-Screener research contact@example.com")
_HEADERS = {"User-Agent": _UA, "Accept-Encoding": "gzip, deflate"}

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_ARCHIVE_DIR = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/"

LOOKBACK_DAYS = 182          # ~6 months of insider activity before the event
BUY_CODES = {"P"}            # open-market purchase (discretionary, conviction)
SELL_CODES = {"S"}           # open-market sale
# A (award/grant), M (option exercise), F (tax withholding), G (gift) etc. are
# NOT discretionary open-market signals, so they are excluded from the signal.

_cik_cache: dict[str, int] = {}


def log(m):
    print(f"[insider] {m}", file=sys.stderr, flush=True)


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


def _val(el, name):
    """A Form-4 amount is usually <name><value>X</value></name>; fall back to text."""
    node = _find(el, name)
    if node is None:
        return None
    v = _find(node, "value")
    txt = (v.text if v is not None and v.text else node.text)
    if txt is None:
        return None
    try:
        return float(str(txt).strip())
    except ValueError:
        return str(txt).strip()


def parse_form4_xml(xml_str: str) -> dict:
    """Parse one Form-4 ownership XML into a role + a list of dated transactions.

    Returns {"is_director","is_officer","is_tenpct", "transactions":[{date, code,
    acq_disp, shares, price, value}]}. Only non-derivative transactions with a
    parseable transaction code are returned; derivative rows are ignored (we want
    common-stock open-market activity). Robust to namespaces and missing fields.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return {"is_director": False, "is_officer": False, "is_tenpct": False,
                "transactions": []}

    def flag(name):
        t = _text(root, name)
        return str(t).strip() in ("1", "true", "True") if t is not None else False

    role = {"is_director": flag("isDirector"),
            "is_officer": flag("isOfficer"),
            "is_tenpct": flag("isTenPercentOwner")}

    txns = []
    for node in root.iter():
        if _strip_ns(node.tag) != "nonDerivativeTransaction":
            continue
        code = _text(node, "transactionCode")
        if not code:
            continue
        d = _text(node, "transactionDate") or _val(node, "transactionDate")
        shares = _val(node, "transactionShares")
        price = _val(node, "transactionPricePerShare")
        acq = _text(node, "transactionAcquiredDisposedCode") or \
            _val(node, "transactionAcquiredDisposedCode")
        shares = shares if isinstance(shares, (int, float)) else None
        price = price if isinstance(price, (int, float)) else None
        value = (shares * price) if (shares is not None and price is not None) else None
        txns.append({"date": str(d)[:10] if d else None, "code": str(code).strip().upper(),
                     "acq_disp": (str(acq).strip().upper() if acq else None),
                     "shares": shares, "price": price, "value": value})
    return {**role, "transactions": txns}


def classify_window(filings: list[dict], event_date: str, lookback_days: int = LOOKBACK_DAYS) -> dict:
    """Aggregate parsed Form-4 filings into an insider signal for one event.

    `filings` = list of {"filing_date","parsed"} where parsed is a parse_form4_xml
    result. Counts only BUY_CODES / SELL_CODES transactions whose TRANSACTION date
    is within (event-lookback, event] AND whose filing was public by the event.
    Director/officer buys are tracked separately (the live doubling-up trigger).
    """
    ev = date.fromisoformat(event_date[:10])
    lo = ev - timedelta(days=lookback_days)
    buy_n = sell_n = 0
    buy_val = sell_val = 0.0
    dir_buy_n = dir_sell_n = 0
    buyers, sellers = set(), set()
    for f in filings:
        fd = _to_date(f.get("filing_date"))
        if fd is not None and fd > ev:      # not yet public at event time — exclude
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
                    buyers.add(id(f))
            elif code in SELL_CODES:
                sell_n += 1
                sell_val += val
                if is_dir:
                    dir_sell_n += 1
                    sellers.add(id(f))
    net_val = buy_val - sell_val
    return {
        "insider_buy_n": buy_n, "insider_sell_n": sell_n,
        "insider_buy_val": round(buy_val, 2), "insider_sell_val": round(sell_val, 2),
        "insider_net_val": round(net_val, 2),
        "director_buy": dir_buy_n > 0, "director_sell": dir_sell_n > 0,
        "insider_signal": net_signal(buy_n, sell_n, net_val),
    }


def net_signal(buy_n: int, sell_n: int, net_val: float) -> str:
    """Coarse three-way label used for bucketing forward returns."""
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
        return None


# --------------------------------------------------------------------------
# Network layer (GitHub Actions only) — fetch + cache, then classify per event
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


def _get(url, session, sleep=0.12):
    time.sleep(sleep)                       # stay under SEC's 10 req/s
    r = session.get(url, timeout=30, headers=_HEADERS)
    r.raise_for_status()
    return r


def _ticker_form4_filings(ticker, session, max_filings=400):
    """All Form-4 filings for a ticker: [{filing_date, xml_url}] (newest first)."""
    cik = _cik_cache.get(ticker.upper())
    if cik is None:
        return []
    sub = _get(_SUBMISSIONS_URL.format(cik=cik), session).json()
    rec = sub.get("filings", {}).get("recent", {})
    forms = rec.get("form", [])
    fdates = rec.get("filingDate", [])
    accs = rec.get("accessionNumber", [])
    docs = rec.get("primaryDocument", [])
    out = []
    for i, form in enumerate(forms):
        if str(form).strip() != "4":
            continue
        acc = accs[i] if i < len(accs) else ""
        doc = docs[i] if i < len(docs) else ""
        if not acc:
            continue
        base = _ARCHIVE_DIR.format(cik=cik, acc_nodash=acc.replace("-", ""))
        # primaryDocument is usually the human .xml; if it is an .htm wrapper the
        # underlying ownership XML sits beside it — we try the primary doc first.
        xml_url = base + doc if doc.endswith(".xml") else base + (doc or "")
        out.append({"filing_date": fdates[i] if i < len(fdates) else None,
                    "xml_url": xml_url, "base": base, "doc": doc})
        if len(out) >= max_filings:
            break
    return out


def _fetch_parsed(filing, session):
    """Fetch + parse one Form-4; if the primary doc is not XML, look for an .xml."""
    url = filing["xml_url"]
    try:
        if not url.endswith(".xml"):
            # list the accession folder and grab the ownership xml
            idx = _get(filing["base"], session).text
            import re as _re
            m = _re.findall(r'href="([^"]+\.xml)"', idx)
            cand = [u for u in m if "R" not in os.path.basename(u)[:2]]
            if not cand:
                return None
            url = filing["base"] + os.path.basename(cand[0])
        xml = _get(url, session).text
        parsed = parse_form4_xml(xml)
        return {"filing_date": filing["filing_date"], "parsed": parsed}
    except Exception as e:  # noqa: BLE001
        log(f"  form4 fetch failed: {e!r}")
        return None


def _in_any_window(filing_date, windows) -> bool:
    fd = _to_date(filing_date)
    if fd is None:
        return False
    return any(lo <= fd <= hi for lo, hi in windows)


def insider_signals_for_events(events_by_ticker: dict, max_tickers=0, max_fetches=120_000) -> dict:
    """events_by_ticker: {TICKER: [event_date_iso, ...]}. Returns
    {(TICKER, event_date): classify_window(...)}. US only. Best-effort: any ticker
    that fails simply yields no signal (columns stay null and drop out of buckets).

    Only Form-4s whose FILING date falls inside some event's lookback window are
    fetched (filing ~ transaction + 2 business days), so network scales with the
    insider activity actually near events, not a company's entire Form-4 history.
    A global `max_fetches` cap is a hard backstop against a runaway run.
    """
    import requests
    out = {}
    try:
        _load_cik_map()
    except Exception as e:  # noqa: BLE001
        log(f"CIK map load failed, skipping insider signal: {e!r}")
        return out
    session = requests.Session()
    tickers = list(events_by_ticker)
    if max_tickers:
        tickers = tickers[:max_tickers]
    fetches = 0
    for n, tk in enumerate(tickers):
        try:
            evs = events_by_ticker[tk]
            # a filing is worth fetching if it could hold a trade in some event window
            windows = [(_to_date(e) - timedelta(days=LOOKBACK_DAYS + 8), _to_date(e))
                       for e in evs if _to_date(e)]
            listing = _ticker_form4_filings(tk, session)
            near = [f for f in listing if _in_any_window(f["filing_date"], windows)]
            if fetches + len(near) > max_fetches:
                log(f"insider fetch cap reached at {tk} ({fetches} fetched); rest skipped")
                break
            parsed = [p for p in (_fetch_parsed(f, session) for f in near) if p]
            fetches += len(near)
            for ev in evs:
                out[(tk, ev)] = classify_window(parsed, ev)
            if (n + 1) % 25 == 0:
                log(f"insiders: {n+1}/{len(tickers)} tickers, {fetches} filings fetched")
        except Exception as e:  # noqa: BLE001
            log(f"{tk}: insider signal failed: {e!r}")
    log(f"insider signals for {len(out)} (ticker,event) pairs, {fetches} filings fetched")
    return out
