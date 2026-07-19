"""
Data access layer. Runs on GitHub Actions runners (unrestricted internet).
Dispatches per market for the universe, prices, benchmarks and market caps.

Sources here are for CANDIDATE GENERATION ONLY (Step 1) — non-primary price
data is fine for that. Every company FACT in Steps 2-3 is taken from primary
sources separately (ASX platform / SEC EDGAR + company IR).
"""
from __future__ import annotations

import io
import sys
import time
from typing import Optional

import pandas as pd

from . import config as cfg


def log(msg: str):
    print(f"[data] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Universe — dispatch on mcfg.UNIVERSE. Returns DataFrame[code, name, yahoo].
# --------------------------------------------------------------------------

def get_universe(mcfg, local_fallback: Optional[str] = None) -> pd.DataFrame:
    kind = getattr(mcfg, "UNIVERSE", "asx_directory")
    if kind == "asx_directory":
        return _asx_universe(local_fallback)
    if kind == "sp1500":
        return _sp1500_universe(local_fallback)
    if kind == "us_expanded":
        return _us_expanded_universe(local_fallback)
    raise ValueError(f"Unknown universe source: {kind}")


# ---- ASX: the exchange's own listed-companies directory ------------------

_ASX_DIRECTORY_URLS = [
    "https://www.asx.com.au/asx/research/ASXListedCompanies.csv",
    "https://asx.api.markitdigital.com/asx-research/1.0/companies/directory/file"
    "?access_token=83ff96335c2d45a094df02a206a39ff4",
]


def _asx_universe(local_fallback: Optional[str]) -> pd.DataFrame:
    import requests
    for url in _ASX_DIRECTORY_URLS:
        try:
            log(f"ASX directory: {url[:60]}...")
            r = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = _parse_asx_directory(r.content)
            if df is not None and len(df) > 500:
                log(f"ASX universe: {len(df)} codes")
                return df
        except Exception as e:  # noqa: BLE001
            log(f"  source failed: {e!r}")
    if local_fallback:
        return _local_universe(local_fallback, ".AX")
    raise RuntimeError("Could not obtain ASX universe and no local fallback given.")


def _parse_asx_directory(content: bytes) -> Optional[pd.DataFrame]:
    for skip in (0, 1, 2, 3):
        try:
            df = pd.read_csv(io.BytesIO(content), skiprows=skip)
        except Exception:  # noqa: BLE001
            continue
        cols = {c.strip().lower(): c for c in df.columns}
        code_key = next((cols[k] for k in cols if "code" in k or k == "asx code"), None)
        name_key = next((cols[k] for k in cols if "company" in k or k == "name"), None)
        sec_key = next((cols[k] for k in cols if "gics" in k or "industry" in k or "sector" in k), None)
        if code_key and name_key:
            out = pd.DataFrame({
                "code": df[code_key].astype(str).str.upper().str.strip(),
                "name": df[name_key].astype(str).str.strip(),
                "sector": (df[sec_key].astype(str).str.strip() if sec_key else None),
            })
            out = out[out["code"].str.match(r"^[A-Z0-9]{3,4}$", na=False)]
            out["yahoo"] = out["code"] + ".AX"
            return out.drop_duplicates("code").reset_index(drop=True)
    return None


# ---- US: the S&P 1500 (S&P 500 + 400 + 600), from Wikipedia --------------

_SP_PAGES = [
    ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", "Security"),
    ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol", "Security"),
    ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol", "Company"),
]


def _sp1500_universe(local_fallback: Optional[str]) -> pd.DataFrame:
    import requests
    frames = []
    for url, sym_col, name_col in _SP_PAGES:
        try:
            html = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"}).text
            tables = pd.read_html(io.StringIO(html))
            # pick the constituents table = the one carrying the symbol column
            for t in tables:
                cols = {str(c).strip().lower(): c for c in t.columns}
                sc = next((cols[k] for k in cols if k in ("symbol", "ticker symbol", "ticker")), None)
                nc = next((cols[k] for k in cols if k in ("security", "company", "name")), None)
                gc = next((cols[k] for k in cols if "gics sector" in k or k == "sector"), None)
                if sc is not None and len(t) > 50:
                    sub = pd.DataFrame({
                        "code": t[sc].astype(str).str.upper().str.strip(),
                        "name": (t[nc].astype(str).str.strip() if nc is not None else t[sc].astype(str)),
                        "sector": (t[gc].astype(str).str.strip() if gc is not None else None),
                    })
                    frames.append(sub)
                    log(f"US {url.split('List_of_')[1][:12]}: {len(sub)} names")
                    break
        except Exception as e:  # noqa: BLE001
            log(f"  S&P page failed ({url[-20:]}): {e!r}")
    if not frames:
        if local_fallback:
            return _local_universe(local_fallback, "")
        raise RuntimeError("Could not obtain US universe (S&P 1500) and no local fallback given.")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["code"].str.match(r"^[A-Z][A-Z0-9.\-]{0,6}$", na=False)]
    # Yahoo uses '-' where the ticker has a class suffix (BRK.B -> BRK-B)
    out["yahoo"] = out["code"].str.replace(".", "-", regex=False)
    out = out.drop_duplicates("code").reset_index(drop=True)
    out["exchange"] = "SP1500"
    log(f"US universe (S&P 1500): {len(out)} names")
    return out


# ---- US: Nasdaq-listed names (widens the S&P 1500 with the extra tech/growth
#          companies that throw off the most overreactions). Nasdaq's own screener
#          API carries symbol/name/sector/marketCap, so the favoured filter works.

_NASDAQ_URL = ("https://api.nasdaq.com/api/screener/stocks"
               "?tableonly=true&limit=25000&offset=0&exchange=NASDAQ")
_NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_cap(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).replace("$", "").replace(",", "").strip()
    try:
        return float(s) if s and s not in ("N/A", "--") else None
    except ValueError:
        return None


def _nasdaq_universe(min_cap: float) -> pd.DataFrame:
    import re
    import requests
    r = requests.get(_NASDAQ_URL, timeout=60, headers=_NASDAQ_HEADERS)
    r.raise_for_status()
    rows = (r.json().get("data") or {}).get("rows") or []
    recs = []
    for row in rows:
        sym = str(row.get("symbol", "")).upper().strip()
        if not re.match(r"^[A-Z][A-Z.]{0,6}$", sym):          # common stock only (no ^, /, units)
            continue
        cap = _parse_cap(row.get("marketCap"))
        if min_cap and cap is not None and cap < min_cap:      # bound the pull to the size floor
            continue
        sec = str(row.get("sector") or "").strip()
        recs.append({"code": sym, "name": str(row.get("name", "")).strip(),
                     "sector": sec or None, "yahoo": sym.replace(".", "-"), "exchange": "NASDAQ"})
    df = pd.DataFrame(recs)
    log(f"Nasdaq screener: {len(df)} common stocks >= size floor")
    return df


def _us_expanded_universe(local_fallback: Optional[str]) -> pd.DataFrame:
    """S&P 1500 UNION Nasdaq-listed (>= size floor), deduped by ticker. Nasdaq-only
    names carry exchange='NASDAQ'; the rest 'SP1500'. Best-effort: if Nasdaq fails,
    falls back to S&P 1500 alone so the run still completes."""
    sp = _sp1500_universe(local_fallback)
    try:
        nq = _nasdaq_universe(cfg.MARKETS["US"]["min_market_cap"])
    except Exception as e:  # noqa: BLE001
        log(f"Nasdaq universe failed ({e!r}); using S&P 1500 only")
        return sp
    if nq.empty:
        return sp
    add = nq[~nq["code"].isin(set(sp["code"]))]
    out = pd.concat([sp, add], ignore_index=True).drop_duplicates("code").reset_index(drop=True)
    log(f"US expanded universe: {len(sp)} S&P + {len(add)} Nasdaq-only = {len(out)} names")
    return out


def _local_universe(path: str, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    code_col = "code" if "code" in df.columns else df.columns[0]
    name_col = "name" if "name" in df.columns else df.columns[min(1, len(df.columns) - 1)]
    out = pd.DataFrame({"code": df[code_col].astype(str).str.upper().str.strip(),
                        "name": df[name_col].astype(str).str.strip()})
    out["yahoo"] = (out["code"] + suffix) if suffix else out["code"].str.replace(".", "-", regex=False)
    return out.drop_duplicates("code").reset_index(drop=True)


# --------------------------------------------------------------------------
# Prices, benchmarks, market caps (market-agnostic; operate on yahoo tickers)
# --------------------------------------------------------------------------

def download_prices(yahoo_tickers: list[str], period_days: int = cfg.HISTORY_CALENDAR_DAYS
                    ) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    batch = 200
    period = f"{max(period_days, 120)}d"
    for i in range(0, len(yahoo_tickers), batch):
        chunk = yahoo_tickers[i:i + batch]
        for attempt in range(3):
            try:
                log(f"prices {i}-{i+len(chunk)} of {len(yahoo_tickers)} (try {attempt+1})")
                data = yf.download(chunk, period=period, interval="1d",
                                   group_by="ticker", auto_adjust=False,
                                   threads=True, progress=False)
                _unpack(data, chunk, out)
                break
            except Exception as e:  # noqa: BLE001
                log(f"  batch failed: {e!r}")
                time.sleep(5 * (attempt + 1))
    log(f"prices obtained for {len(out)}/{len(yahoo_tickers)} tickers")
    return out


def _unpack(data: pd.DataFrame, chunk: list[str], out: dict):
    if data is None or len(data) == 0:
        return
    for t in chunk:
        try:
            sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = sub[["Close", "Volume"]].dropna(how="all")
            if len(df) >= cfg.MIN_VOL_OBS:
                out[t] = df
        except Exception:  # noqa: BLE001
            continue


def download_benchmarks(mcfg, period_days: int = cfg.HISTORY_CALENDAR_DAYS) -> dict[str, pd.Series]:
    import yfinance as yf
    out = {}
    for b in (mcfg.BENCHMARK_200, mcfg.BENCHMARK_300):
        try:
            d = yf.download(b, period=f"{max(period_days,120)}d", interval="1d",
                            auto_adjust=False, progress=False)
            out[b] = d["Close"].dropna().squeeze()
        except Exception as e:  # noqa: BLE001
            log(f"benchmark {b} failed: {e!r}")
    return out


def get_market_caps(yahoo_tickers: list[str]) -> dict[str, float]:
    import yfinance as yf
    caps: dict[str, float] = {}
    for t in yahoo_tickers:
        for attempt in range(2):
            try:
                fi = yf.Ticker(t).fast_info
                cap = getattr(fi, "market_cap", None) or fi.get("market_cap")
                if cap:
                    caps[t] = float(cap)
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
    return caps
