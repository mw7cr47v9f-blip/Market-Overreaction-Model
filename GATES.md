# Cragent.ai — Approved Gate Set (FROZEN)

**Status: APPROVED by Keegan.** This is the single source of truth for the model's
gates. Every backtest run (all 8 exchanges) and the live daily screen are computed
against THIS set. Nothing is added, removed, or loosened without an explicit,
recorded change to this file.

The 8 exchanges: **NYSE, NASDAQ** (market=US) · **ASX · TSX · LSE · TSE · SGX · SEHK**.

---

## The gates (in order)

| # | Gate | Threshold | Notes |
|---|------|-----------|-------|
| 1 | **Drop** | raw decline ≥ **22%** | Approved 2026-07 (was 20%). Deeper = bigger overreaction. |
| 2 | **Collapse floor** | raw decline ≤ 60% | Worse than −60% in ≤5 days = solvency/delisting, not overreaction. |
| 3 | **Dislocation (z)** | z ≤ −2.5 | Drop ≥ 2.5 SD below the stock's own pre-drop volatility. Near-slack once the other gates apply (median gated z ≈ −7); **kept as a low-cost noise filter, under review** — the re-run loosens it at emission so removal can be tested. |
| 4 | **Index-relative** | ≤ −10pp vs benchmark | The drop is stock-specific, not a market-wide selloff. |
| 5 | **Liquidity** | avg daily value ≥ per-market floor | US $2M/day. See per-market table. |
| 6 | **Market cap** | ≥ per-market floor | **US $1bn.** See per-market table. Enforced live; back-filled into the backtest. |
| 7 | **Sector** | NOT in avoid-list | Avoid regex: `material \| pharma \| biotech \| telecom \| communication \| real estate`. |
| 8 | **Quality** | NPAT margin ≥ −5% **AND** FCF margin ≥ 0 | Latest annual (10-K / point-in-time). |
| 9 | **Structural trigger** | EXCLUDE contract_loss / cost_impair / distress | **Option A — APPLIED.** From SEC 8-K item codes (US only; non-US pass by analogy). Earnings/guidance drops are kept. |
| 10 | **Director buy** | ≥ **$50,000** on-market, prior ~6 months | SEC Form 4 (US only; non-US pass by analogy). The "real-money" test. |
| 11 | **Value gate** | pre-drop P/E ≤ 1.54 × sector median | Fails OPEN on missing/negative earnings. Inert in ablation but kept as a guardrail. |
| 12 | **Hold** | fixed 3-month, no stop-loss | Time-based exit. |

Not gates (for the record): **conviction weighting** is a position-sizing overlay
(tested ≈ neutral, slightly negative); **trading-halt** is a live-verification check
in the daily report, never a backtest filter.

## Per-market floors

| Exchange | Market cap floor | Liquidity floor / day |
|----------|------------------|-----------------------|
| US (NYSE, NASDAQ) | US$1,000,000,000 | US$2,000,000 |
| ASX | A$100,000,000 | A$150,000 |
| TSX | C$200,000,000 | C$1,000,000 |
| LSE | £100,000,000 | £500,000 |
| TSE | ¥20,000,000,000 | ¥50,000,000 |
| SGX | S$200,000,000 | S$300,000 |
| SEHK | HK$1,000,000,000 | HK$3,000,000 |

## Where each gate is enforced

- **Live daily screen** (`run.py` + `config.py`): ALL 12 gates. Verified.
- **Backtest emission** (`backtest_factor.py`, always-on at scan): collapse floor,
  index-relative, liquidity, drop ≥10% (loose, for toggling). *z will move OUT of
  emission in the re-run so it can be toggled.*
- **Backtest analysis / enrichment** (applied to the events): drop ≥22%, sector,
  quality, director ≥$50k, value, market cap, structural trigger.

## Known enrichment status (as of approval)

The existing NYSE + NASDAQ event files were gated WITHOUT market cap or structural
triggers (the columns weren't emitted). Both are being back-filled onto the existing
files — **no full re-run required**:

- **Structural triggers**: `sec_triggers.enrich()` (SEC 8-K, free) → `trigger_primary`.
- **Market cap**: targeted EODHD fetch (shares × event-date price) → `market_cap`.

Only **z-loosening** requires a fresh run, folded into the eventual clean all-8-market
re-run that emits `market_cap`, `avg_daily_value`, and `trigger_primary` natively.

## Toggle protocol

Toggling begins ONLY after all 8 markets are computed on the set above. Candidates the
user has flagged to toggle later: market cap, liquidity, dislocation (z), drop floor.
Any toggle is a recorded experiment against this frozen baseline — the baseline does
not change unless this file is updated and re-approved.

---
*Last approved: 2026-07 · Change log below.*

- 2026-07: Initial freeze. Drop floor 20%→22%. Market cap + structural triggers
  confirmed in-scope after audit found them missing from prior backtest numbers.
