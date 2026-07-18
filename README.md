# ASX overreaction screen — Step 1 engine

This repo runs a **precise, daily** statistical screen over the whole ASX
(all listed codes, ≥ $100m market cap) to find quality-candidate stocks that
dropped sharply and *abnormally* over a rolling window of 5 trading days or
less. It runs itself on GitHub Actions after each ASX close and commits the
results back here, where the daily Claude task reads them for the
primary-source analysis (Steps 2–3).

## Why it runs here and not inside Claude

The Claude cloud session can't reach any market-data source (its network
allowlist blocks Yahoo, Stooq, the ASX price API and every data vendor).
GitHub Actions runners have normal internet, so the data pull + the exact
maths run here; GitHub is one of the few hosts Claude *can* read from, so it
picks up the committed results. That split is what makes a precise, hands-off
daily feed possible.

## What it computes (exactly, not approximately)

For every stock, on each trading day, for every window length 1–5 days:

1. **Statistical extremity** — window return ≤ **−2.5σ**, where σ is the
   stock's own trailing daily volatility (last ~90 trading days, taken strictly
   *before* the window so the crash can't inflate its own baseline) scaled by
   √(window length).
2. **Index-relative** — underperforms its benchmark over the same window by
   ≥ **10 percentage points** (ASX 200 for ≥ $5bn names, ASX 300 otherwise),
   so broad selloffs are excluded.
3. **Absolute floor** — raw decline ≥ **10%**.

Plus a **liquidity floor** (avg daily traded value ≥ **$150k** over ~90 days)
and the **≥ $100m market-cap** gate.

A stock is flagged if *any* window qualifies; the most statistically extreme
window is reported. Each crash is anchored to its worst single day and logged
**once** — the rolling window shifting day-to-day does not re-flag it.

All thresholds live in `screener/config.py` — the knobs to turn if the feed is
too noisy (tighten `Z_THRESHOLD` to −3.0) or too quiet.

## Outputs (committed to `data/`)

- `candidates_new.json` — just today's genuinely-new events (the alert).
- `candidates_all.csv` — append-only running log, one row per event, with a
  `status` column (`New` → filled in later by the analysis step).
- `state.json` — dedup memory (ticker + event-date keys already seen).
- `announcements/<TICKER>_<date>.json` — ASX announcements around each new
  event (headline, price-sensitive flag, PDF link) as a starting point for the
  primary-source analysis.

## One-time setup (~10 minutes)

1. Sign in to GitHub and create a **new repository** — name it e.g.
   `asx-overreaction`. Private is fine. Don't add a README (this repo has one).
2. Upload these files, preserving the folder layout:
   ```
   requirements.txt
   README.md
   screener/            (all .py files)
   .github/workflows/daily-screen.yml
   data/                (can start empty; the job creates files here)
   ```
   Easiest: on the repo page, **Add file → Upload files**, drag the whole
   folder in, commit. (Or `git clone` then copy the files in and push.)
3. Go to the **Settings → Actions → General**, and under *Workflow
   permissions* choose **Read and write permissions** (so the daily job can
   commit results). Save.
4. Go to the **Actions** tab, enable workflows if prompted, pick
   *ASX overreaction daily screen*, and click **Run workflow** to do the first
   run by hand. It will build the universe, run the screen, and commit the
   first `data/` files. Check the run log for the summary table.
5. After that it runs automatically at **08:00 UTC, Mon–Fri** (after the Sydney
   close). Nothing further needed.

If the ASX directory endpoint ever fails, drop a `data/universe.csv` with
`code,name` columns into the repo and the job will use it as the universe.

## Local check (optional)

```
pip install -r requirements.txt
python -m screener.run --self-test          # offline logic check
python -m screener.run --data-dir data --limit 50   # tiny live run
```

## How the analysis half connects

The daily Claude task pulls this repo, reads `candidates_new.json` and the
matching `announcements/*.json`, then does the quality gate and the
"was the drop justified?" analysis **from primary sources only** (the
company's own results releases + announcements.asx.com.au), appends verdicts
to the running spreadsheet, and sends you a compact summary of just the new
names that cleared the gate.
