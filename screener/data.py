"""
Data access layer. Runs on GitHub Actions runners, which have unrestricted
internet (unlike the Claude cloud sandbox). Nothing here is expected to work
from inside a Claude session — that's the whole reason the screen runs on
Actions and commits results back for Claude to read.

Sources used here are for CANDIDATE GENERATION ONLY (Step 1). Per the brief,
non-primary price data is fine for generating the candidate list; every factual
claim about a company in Steps 2-3 is made from primary sources separately.
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
# Universe: every ASX-listed code. The >= $100m cap filter is applied later,
# only to price-qualifying names, so here we just need the full code list.
# --------------------------------------------------------------------------

_ASX_DIRECTORY_URLS = [
    # ASX's own listed-companies file has moved over the years; try the known
    # locations in order. If all fail, we fall back to a committed universe.csv.
    "https://www.asx.com.au/asx/research/ASXListedCompanies.csv",
    "https://asx.api.markitdigital.com/asx-research/1.0/companies/directory/file"
    "?access_token=83ff96335c2d45a094df02a206a39ff4",
]


def get_universe(local_fallback: Optional[str] = None) -> pd.DataFrame:
    """Return DataFrame[code, name, yahoo]. code = plain ASX code (e.g. 'CBA')."""
    import requests

    for url in _ASX_DIRECTORY_URLS:
        try:
            log(f"fetching ASX directory: {url[:60]}...")
            r = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = _parse_directory(r.content)
            if df is not None and len(df) > 500:
                log(f"universe: {len(df)} codes from ASX directory")
                return df
        except Exception as e:  # noqa: BLE001
            log(f"  directory source failed: {e!r}")

    if local_fallback:
        log(f"falling back to committed universe file: {local_fallback}")
        df = pd.read_csv(local_fallback)
        df.columns = [c.strip().lower() for c in df.columns]
        code_col = "code" if "code" in df.columns else df.columns[0]
        name_col = "name" if "name" in df.columns else df.columns[1]
        out = pd.DataFrame({"code": df[code_col].astype(str).str.upper().str.strip(),
                            "name": df[name_col].astype(str).str.strip()})
        out["yahoo"] = out["code"] + ".AX"
        return out.drop_duplicates("code").reset_index(drop=True)

    raise RuntimeError("Could not obtain ASX universe from any source and no local fallback given.")


def _parse_directory(content: bytes) -> Optional[pd.DataFrame]:
    """Handle both the old CSV layout and the markitdigital CSV export."""
    for skip in (0, 1, 2, 3):
        try:
            df = pd.read_csv(io.BytesIO(content), skiprows=skip)
        except Exception:  # noqa: BLE001
            continue
        cols = {c.strip().lower(): c for c in df.columns}
        code_key = next((cols[k] for k in cols if "code" in k or k == "asx code"), None)
        name_key = next((cols[k] for k in cols if "company" in k or k == "name"), None)
        if code_key and name_key:
            out = pd.DataFrame({
                "code": df[code_key].astype(str).str.upper().str.strip(),
                "name": df[name_key].astype(str).str.strip(),
            })
            out = out[out["code"].str.match(r"^[A-Z0-9]{3,4}$", na=False)]
            out["yahoo"] = out["code"] + ".AX"
            return out.drop_duplicates("code").reset_index(drop=True)
    return None


# --------------------------------------------------------------------------
# Prices: one bulk download for the whole universe + the benchmark indices.
# --------------------------------------------------------------------------

def download_prices(yahoo_tickers: list[str], period_days: int = cfg.HISTORY_CALENDAR_DAYS
                    ) -> dict[str, pd.DataFrame]:
    """Return {yahoo_ticker: DataFrame[Close, Volume]} indexed by date.

    Uses yfinance in batches with retries. Missing/failed tickers are skipped.
    """
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
            if isinstance(data.columns, pd.MultiIndex):
                sub = data[t]
            else:
                sub = data  # single ticker
            df = sub[["Close", "Volume"]].dropna(how="all")
            if len(df) >= cfg.MIN_VOL_OBS:
                out[t] = df
        except Exception:  # noqa: BLE001
            continue


def download_benchmarks(period_days: int = cfg.HISTORY_CALENDAR_DAYS) -> dict[str, pd.Series]:
    import yfinance as yf
    out = {}
    for b in (cfg.BENCHMARK_200, cfg.BENCHMARK_300):
        try:
            d = yf.download(b, period=f"{max(period_days,120)}d", interval="1d",
                            auto_adjust=False, progress=False)
            out[b] = d["Close"].dropna().squeeze()
        except Exception as e:  # noqa: BLE001
            log(f"benchmark {b} failed: {e!r}")
    return out


def get_market_caps(yahoo_tickers: list[str]) -> dict[str, float]:
    """Fetch market cap (AUD) for a SMALL shortlist of price-qualifying names.
    Only called on the handful that clear the price screen, so per-ticker
    .fast_info calls are cheap."""
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
