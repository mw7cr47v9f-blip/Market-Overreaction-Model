"""
Pure statistical core of the overreaction screen.

No I/O here — every function takes plain pandas objects and returns plain data,
so the logic can be unit-tested against synthetic series with known answers
(see run.py --self-test). This is where correctness actually matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Candidate:
    market: str
    currency: str
    ticker: str
    name: str
    market_cap: float
    benchmark: str
    # the most statistically extreme qualifying window:
    window_len: int
    window_start: str      # ISO date (first day of the decline window)
    window_end: str        # ISO date (= run date)
    event_date: str        # ISO date of the single worst day inside the window
    raw_return: float      # window return, e.g. -0.18
    index_return: float    # benchmark window return
    index_relative: float  # raw_return - index_return
    z_score: float         # raw_return / (daily_sigma * sqrt(window_len))
    daily_sigma: float     # trailing daily volatility used
    avg_daily_value: float # trailing average traded value (AUD)
    last_close: float
    # also record the single largest-drop window for the economics narrative:
    max_drop_window_len: int
    max_drop_return: float

    def key(self) -> str:
        """Dedup identity: a given crash day for a given stock is ONE event,
        no matter how many rolling windows still contain it on later days.
        Namespaced by market so an ASX and a US ticker can't collide."""
        return f"{self.market}:{self.ticker}:{self.event_date}"

    def to_dict(self) -> dict:
        return asdict(self)


def daily_returns(close: pd.Series) -> pd.Series:
    return close.sort_index().pct_change()


def _window_return(close: pd.Series, i_end: int, w: int) -> Optional[float]:
    i_start = i_end - w
    if i_start < 0:
        return None
    c0 = close.iloc[i_start]
    c1 = close.iloc[i_end]
    if c0 is None or c1 is None or c0 <= 0 or np.isnan(c0) or np.isnan(c1):
        return None
    return c1 / c0 - 1.0


def evaluate_series(
    ticker: str,
    name: str,
    close: pd.Series,
    volume: pd.Series,
    bench_close: pd.Series,
    market_cap: float,
    cfg,
) -> Optional[Candidate]:
    """
    Evaluate one stock on its latest trading day. Returns a Candidate if any
    window of length 1..max qualifies on all three conditions, else None.

    `close`, `volume` are indexed by trading-day Timestamp (ascending).
    `bench_close` is the benchmark index close, indexed by Timestamp.
    `market_cap` in AUD (pre-computed by the caller).
    """
    close = close.dropna().sort_index()
    if len(close) < cfg.MIN_VOL_OBS + max(cfg.WINDOW_LENGTHS) + 2:
        return None

    # Size filter.
    if market_cap is None or market_cap < cfg.MIN_MARKET_CAP:
        return None

    # Liquidity filter: average traded value over the trailing vol window.
    traded_value = (close * volume.reindex(close.index)).dropna()
    adv = float(traded_value.tail(cfg.VOL_LOOKBACK).mean()) if len(traded_value) else 0.0
    if adv < cfg.MIN_AVG_DAILY_VALUE:
        return None

    rets = daily_returns(close)
    i_end = len(close) - 1
    run_date = close.index[i_end]

    # Align benchmark onto the stock's trading calendar.
    bench = bench_close.dropna().sort_index().reindex(close.index).ffill()

    qualifying = []
    max_drop = (None, 0.0)  # (window_len, return) — most negative raw drop seen

    for w in cfg.WINDOW_LENGTHS:
        r = _window_return(close, i_end, w)
        if r is None:
            continue
        if r < max_drop[1]:
            max_drop = (w, r)

        # Trailing daily vol from returns STRICTLY BEFORE the window starts, so
        # the crash cannot inflate its own baseline.
        i_win_start = i_end - w
        baseline = rets.iloc[:i_win_start].dropna().tail(cfg.VOL_LOOKBACK)
        if len(baseline) < cfg.MIN_VOL_OBS:
            continue
        daily_sigma = float(baseline.std(ddof=1))
        if daily_sigma <= 0 or math.isnan(daily_sigma):
            continue
        window_sigma = daily_sigma * math.sqrt(w)
        z = r / window_sigma

        # Benchmark return over the same window.
        b0, b1 = bench.iloc[i_win_start], bench.iloc[i_end]
        if b0 is None or np.isnan(b0) or b0 <= 0:
            continue
        bench_r = b1 / b0 - 1.0
        index_rel = r - bench_r

        cond1 = z <= cfg.Z_THRESHOLD
        cond2 = index_rel <= cfg.INDEX_REL_THRESHOLD
        cond3 = r <= cfg.ABS_DROP_THRESHOLD
        if cond1 and cond2 and cond3:
            # worst single day inside the window -> event anchor
            win_rets = rets.iloc[i_win_start + 1: i_end + 1]
            event_date = win_rets.idxmin() if len(win_rets) else close.index[i_win_start + 1]
            qualifying.append(dict(
                window_len=w,
                window_start=close.index[i_win_start],
                event_date=event_date,
                raw_return=r,
                index_return=bench_r,
                index_relative=index_rel,
                z_score=z,
                daily_sigma=daily_sigma,
            ))

    if not qualifying:
        return None

    # Headline window = most statistically extreme (lowest z).
    best = min(qualifying, key=lambda q: q["z_score"])

    return Candidate(
        market=getattr(cfg, "MARKET", "ASX"),
        currency=getattr(cfg, "CURRENCY", "AUD"),
        ticker=ticker,
        name=name,
        market_cap=float(market_cap),
        benchmark=(cfg.BENCHMARK_200 if market_cap >= cfg.LARGE_CAP_CUTOFF else cfg.BENCHMARK_300),
        window_len=best["window_len"],
        window_start=best["window_start"].date().isoformat(),
        window_end=run_date.date().isoformat(),
        event_date=best["event_date"].date().isoformat(),
        raw_return=round(best["raw_return"], 6),
        index_return=round(best["index_return"], 6),
        index_relative=round(best["index_relative"], 6),
        z_score=round(best["z_score"], 4),
        daily_sigma=round(best["daily_sigma"], 6),
        avg_daily_value=round(adv, 2),
        last_close=round(float(close.iloc[i_end]), 6),
        max_drop_window_len=int(max_drop[0]) if max_drop[0] else best["window_len"],
        max_drop_return=round(float(max_drop[1]), 6),
    )
