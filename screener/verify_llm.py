"""
LLM verification + dashboard builder for the Cragent.ai daily screen.

Runs in GitHub Actions AFTER screener.run has written the day's outputs and
BEFORE the commit + email steps. For each confirmed-entry BUY and each new
qualifying drop it:

  * reads the pre-fetched SEC filing links (data/announcements/US_<TICKER>_*.json)
    and fetches the primary 8-K / 10-Q text from EDGAR (same SEC User-Agent the
    screen already uses);
  * calls the Anthropic Messages API (raw urllib -- no SDK dependency) to produce
    the JUDGEMENT layer the mechanical screen cannot: a trigger classification
    (quantifiable vs judgement), a plain-language read, implied-vs-actual, an
    insider-activity narrative, and a going-concern / halt check from the filing;
  * assembles the dashboard DATA object and renders dashboard_template.html into
    data/daily_dashboard.html, and writes data/verification.json for the email.

The quality-gate numbers (NPAT margin, FCF) and the director-buy size are taken
straight from what the screen already computed -- they are NOT re-derived by the
model, so those facts stay deterministic. The model only reads the trigger filing
and writes the narrative on top.

FAIL-SOFT BY DESIGN. If ANTHROPIC_API_KEY is unset, or any fetch / API call
fails, it writes a MECHANICAL dashboard (structured screen facts only, no
judgement) and a minimal verification.json, then exits 0. The daily email and its
3-month SELL reminders must NEVER be blocked by this step.

Model: defaults to claude-sonnet-5; override with CRAGENT_LLM_MODEL (e.g.
claude-haiku-4-5 for lower cost, claude-opus-4-8 for maximum judgement).

Not financial advice -- see the DISCLAIMER in email_report.py.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
import urllib.error
from datetime import date

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"
_MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                "Oct", "Nov", "Dec"]
_MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"]


def log(m):
    print(f"[verify] {m}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# small formatters
# --------------------------------------------------------------------------- #
def _nice(iso):
    """'2026-07-08' -> '8 Jul 2026'. Passes through anything unparseable."""
    try:
        d = date.fromisoformat(str(iso)[:10])
        return f"{d.day} {_MONTHS_ABBR[d.month - 1]} {d.year}"
    except Exception:  # noqa: BLE001
        return str(iso)


def _nice_full(iso):
    try:
        d = date.fromisoformat(str(iso)[:10])
        return f"{d.day} {_MONTHS_FULL[d.month - 1]} {d.year}"
    except Exception:  # noqa: BLE001
        return str(iso)


def _human_money(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    a = abs(v)
    if a >= 1e9:
        return f"${v/1e9:.2f}bn" if a < 1e10 else f"${v/1e9:.1f}bn"
    if a >= 1e6:
        return f"${v/1e6:.0f}m"
    if a >= 1e3:
        return f"${v/1e3:.0f}k"
    return f"${v:.0f}"


def _pct(v, suffix="%"):
    try:
        return f"{float(v)*100:+.1f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _znum(v):
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- #
# primary-source filing text
# --------------------------------------------------------------------------- #
def _announcement(data_dir, ticker, event_date):
    """The pre-fetched SEC record for this drop, or {}."""
    path = os.path.join(data_dir, "announcements", f"US_{ticker}_{event_date}.json")
    return _load(path)


def _strip_html(html):
    """Best-effort HTML -> text. lxml if present, else a crude regex."""
    if not html:
        return ""
    try:
        import lxml.html as LH
        return LH.fromstring(html).text_content()
    except Exception:  # noqa: BLE001
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text)


def _fetch_filing_text(url, limit=18000):
    """Fetch a primary EDGAR document and return trimmed plain text. '' on any
    failure -- the model then reasons from the metrics alone."""
    if not url:
        return ""
    try:
        import requests
        from . import us_announcements
        r = requests.get(url, headers=us_announcements._HEADERS, timeout=30)
        if r.status_code != 200:
            return ""
        text = _strip_html(r.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit]
    except Exception as e:  # noqa: BLE001
        log(f"filing fetch failed for {url}: {e!r}")
        return ""


def _sources_from(ann):
    """[[label, url], ...] from the pre-fetched filing list."""
    out = []
    for f in (ann.get("filings") or []):
        u = f.get("filing_url")
        if not u:
            continue
        lbl = f"{f.get('form', 'Filing')} ({_nice(f.get('date'))}) — SEC EDGAR"
        out.append([lbl, u])
    return out


def _best_filing_url(ann):
    """Prefer an 8-K (the event trigger), else the first filing with a URL."""
    filings = [f for f in (ann.get("filings") or []) if f.get("filing_url")]
    if not filings:
        return ""
    for f in filings:
        if str(f.get("form", "")).startswith("8-K"):
            return f["filing_url"]
    return filings[0]["filing_url"]


# --------------------------------------------------------------------------- #
# quality gate + insider (deterministic, from what the screen already computed)
# --------------------------------------------------------------------------- #
def _gate_stack(cand, llm):
    """The FULL model-gate checklist for a card. Every threshold gate shows as a pass
    tick because the name is in candidates_new — it cleared the whole stack to get there.
    NPAT/FCF come from the screen's companyfacts margins (fail-open, ok=1, when a margin
    is unavailable, so we never contradict the screen). Structural-trigger and halt come
    from the model's filing read; 'not checked' on the mechanical (no-LLM) path."""
    npat, fcf = cand.get("npat_margin"), cand.get("fcf_margin")
    npat_ok = 1 if (npat is None or float(npat) >= -0.05) else 0
    fcf_ok = 1 if (fcf is None or float(fcf) >= 0.0) else 0
    npat_txt = _pct(npat) if npat is not None else "per screen"
    fcf_txt = _pct(fcf) if fcf is not None else "per screen"
    dval = cand.get("director_buy_val")
    dtxt = f"${float(dval):,.0f}" if dval else "confirmed"
    stack = [
        ["Drop ≥20%", _pct(cand.get("raw_return")), 1],
        ["Stock-specific ≤−10pp", _pct(cand.get("index_relative"), "pp"), 1],
        ["Dislocation z ≤−2.5", _znum(cand.get("z_score")), 1],
        ["Liquidity floor", _human_money(cand.get("avg_daily_value")) + "/day", 1],
        ["Broad sector", cand.get("sector") or "broad", 1],
        ["NPAT margin ≥−5%", npat_txt, npat_ok],
        ["Free cash flow ≥0", fcf_txt, fcf_ok],
    ]
    if llm:
        gc = llm.get("going_concern") or {}
        ht = llm.get("halt") or {}
        stack += [["No structural trigger", str(gc.get("text", "clear")), 1 if gc.get("ok", 1) else 0],
                  ["Not halted", str(ht.get("text", "no")), 1 if ht.get("ok", 1) else 0]]
    else:
        stack += [["No structural trigger", "not checked", 1],
                  ["Not halted", "not checked", 1]]
    stack.append(["Director buy ≥$50k", dtxt, 1])
    return stack


def _insider_text(cand):
    val = cand.get("director_buy_val")
    if val:
        try:
            return (f"Prior-6-month on-market director buying of ${float(val):,.0f} "
                    f"(SEC Form 4) — clears the $50k real-money test.")
        except (TypeError, ValueError):
            pass
    if cand.get("director_buy"):
        return "Prior-6-month on-market director buying confirmed (SEC Form 4)."
    return "Director-buy status per the screen's Form 4 gate."


# --------------------------------------------------------------------------- #
# Anthropic Messages API (raw urllib)
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You are a financial analyst verifying a single oversold US stock for the "
    "Cragent.ai Market Overreaction model. You are given the screen's structured "
    "metrics and, when available, the text of the primary SEC filing that "
    "triggered the drop. Reason ONLY from the provided filing text and metrics; "
    "never invent figures. This is informational analysis, not investment advice. "
    "Return ONLY a single JSON object, no prose and no markdown fences."
)

_INSTRUCTIONS = """Return a JSON object with exactly these keys:
{
 "status": one of "over" (overreaction plausible), "judge" (judgement required),
           "just" (the move looks justified) or "fail" (a red flag emerged);
 "read": 1-3 sentence plain-language read of whether this looks like an
         overreaction and why (reference the cap loss vs the stated impact);
 "trigger": either "(A) Quantifiable" or "(B) Judgement required";
 "detail": one sentence stating what the primary filing actually disclosed
           (the trigger). If no filing text was provided, say so plainly;
 "implied": one sentence on the market-cap lost vs the mechanical impact of the
            news (implied vs actual). "" if not determinable;
 "going_concern": {"text": short phrase e.g. "none noted", "ok": 1 or 0};
 "halt": {"text": short phrase e.g. "no", "ok": 1 or 0}
}
Base going_concern/halt on the filing text if present; if absent, use
{"text":"not stated in filing","ok":1}. Keep every string tight."""


def _anthropic(api_key, model, user_payload, max_tokens=900):
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user_payload}],
    }).encode()
    req = urllib.request.Request(
        API_URL, data=body, method="POST",
        headers={"x-api-key": api_key, "anthropic-version": API_VERSION,
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        js = json.loads(resp.read())
    parts = js.get("content", []) or []
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def _extract_json(text):
    """Pull the first {...} object out of a model reply."""
    if not text:
        return None
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:  # noqa: BLE001
        return None


def _verify_one(cand, ann, api_key, model):
    """LLM judgement for one name -> dict, or None on any failure."""
    filing_url = _best_filing_url(ann)
    filing_text = _fetch_filing_text(filing_url) if filing_url else ""
    metrics = {
        "ticker": cand.get("ticker"), "name": cand.get("name"),
        "sector": cand.get("sector"),
        "drop_pct": _pct(cand.get("raw_return")),
        "vs_index_pp": _pct(cand.get("index_relative"), "pp"),
        "z_score": _znum(cand.get("z_score")),
        "market_cap": _human_money(cand.get("market_cap")),
        "avg_daily_value": _human_money(cand.get("avg_daily_value")) + "/day",
        "npat_margin": cand.get("npat_margin"),
        "fcf_margin": cand.get("fcf_margin"),
        "director_buy_usd": cand.get("director_buy_val"),
        "event_date": cand.get("event_date"),
    }
    payload = (f"{_INSTRUCTIONS}\n\nSCREEN METRICS:\n{json.dumps(metrics, indent=2)}\n\n"
               f"PRIMARY FILING TEXT (may be empty):\n{filing_text or '(none retrieved)'}")
    try:
        raw = _anthropic(api_key, model, payload)
    except urllib.error.HTTPError as e:
        log(f"{cand.get('ticker')}: API HTTP {e.code}: {e.read()[:200]!r}")
        return None
    except Exception as e:  # noqa: BLE001
        log(f"{cand.get('ticker')}: API call failed: {e!r}")
        return None
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        log(f"{cand.get('ticker')}: could not parse JSON from model reply")
        return None
    return obj


# --------------------------------------------------------------------------- #
# card assembly (LLM result -> dashboard candidate + verification record)
# --------------------------------------------------------------------------- #
def _window_str(cand):
    wl = cand.get("window_len")
    a, b = _nice(cand.get("window_start")), _nice(cand.get("window_end"))
    lead = f"{wl} days" if wl else "window"
    return f"{lead} · {a} – {b}"


def _card(cand, ann, llm):
    """Build the dashboard candidate dict (+ the narrative used by the email)."""
    gate = _gate_stack(cand, llm)
    insider = _insider_text(cand)
    sources = _sources_from(ann) or [
        ["SEC EDGAR full-text search",
         f"https://efts.sec.gov/LATEST/search-index?q=%22{cand.get('ticker')}%22"]]

    if llm:
        status = llm.get("status") if llm.get("status") in ("over", "judge", "just", "fail") else "judge"
        read = llm.get("read") or ""
        trigger = llm.get("trigger") or "(B) Judgement required"
        detail = llm.get("detail") or "See the linked SEC filing."
        implied = llm.get("implied") or ""
        # let the model's own insider line win only if it explicitly returned one
        insider = llm.get("insider") or insider
    else:
        status = "judge"
        read = (f"Fell {_pct(cand.get('raw_return'))} versus the market "
                f"({_pct(cand.get('index_relative'), 'pp')}), a "
                f"{_znum(cand.get('z_score'))}σ dislocation. Passed the model's "
                f"quality and director-buy gates. Primary-source verification was "
                f"unavailable this run — open the linked filing before acting.")
        trigger = "(B) Judgement required"
        d0 = ann.get("filings") or []
        detail = (d0[0].get("description") or f"See {d0[0].get('form','the')} filing on SEC EDGAR.") \
            if d0 else "See the linked SEC filing."
        implied = ""

    card = {
        "tk": cand.get("ticker"), "mk": "US", "name": cand.get("name"),
        "sector": cand.get("sector"), "status": status,
        "drop": _pct(cand.get("raw_return")),
        "idx": _pct(cand.get("index_relative"), "pp"),
        "z": _znum(cand.get("z_score")),
        "window": _window_str(cand),
        "cap": _human_money(cand.get("market_cap")),
        "liq": _human_money(cand.get("avg_daily_value")) + "/day",
        "read": read, "trigger": trigger, "detail": detail, "implied": implied,
        "insider": insider, "gate": gate, "sources": sources,
    }
    narrative = {"status": status, "read": read, "trigger": trigger,
                 "detail": detail, "implied": implied, "insider": insider,
                 "sources": sources}
    return card, narrative


# --------------------------------------------------------------------------- #
# ledger + summary
# --------------------------------------------------------------------------- #
def _pos(r, flag=None):
    return {"tk": r.get("ticker"), "mk": r.get("market", "US"),
            "sector": r.get("sector"), "entry_date": _nice(r.get("entry_date")),
            "entry": r.get("entry_price"), "last": r.get("last_price"),
            "ret": r.get("ret_pct"), "held": r.get("days_held"),
            "due": _nice(r.get("sell_due_date")), "flag": flag or r.get("flag") or "OPEN"}


def _ledger(led):
    model = led.get("model", {}) if isinstance(led, dict) else {}
    holdings = led.get("holdings", []) if isinstance(led, dict) else []
    tally = model.get("realised_tally", {}) or {}
    open_rows = [_pos(r, "OPEN") for r in (model.get("open_positions") or [])]
    sells = ([_pos(r, "SELL_DUE") for r in (model.get("closed_this_run") or [])]
             + [_pos(r, "SELL_SOON") for r in (model.get("sell_soon") or [])])
    hold_rows = [_pos(h, h.get("flag")) for h in holdings]
    wr = tally.get("win_rate")
    av = tally.get("avg_return")
    return {
        "open": open_rows, "sells_due": sells,
        "tally": {"n_closed": tally.get("n_closed", 0),
                  "win_rate": (f"{wr*100:.0f}%" if isinstance(wr, (int, float)) else "—"),
                  "avg": (f"{av*100:+.1f}%" if isinstance(av, (int, float)) else "—")},
        "holdings": hold_rows,
    }, len(sells) + sum(1 for h in holdings if str(h.get("flag", "")).upper() in ("SELL_DUE", "SELL_SOON"))


def _summary(new, confirmed, pending, ledger, sells_due_n):
    if new:
        worst = min(new, key=lambda c: c.get("raw_return", 0))
        extreme = {"v": _pct(worst.get("raw_return")),
                   "s": f"{worst.get('ticker')} · {_znum(worst.get('z_score'))}σ"}
    else:
        extreme = {"v": "—", "s": "no new drops"}
    tickers = ", ".join(c.get("ticker") for c in new[:4]) or "none"
    return {
        "flagged": {"v": len(new), "s": "≥20%, broad sectors"},
        "cleared": {"v": len(new), "s": tickers},
        "extreme": extreme,
        "open": {"v": len(ledger["open"]), "s": "model book"},
        "sells_due": {"v": sells_due_n, "s": ("action today" if sells_due_n else "none today")},
        "win_rate": {"v": ledger["tally"]["win_rate"], "s": f"{ledger['tally']['n_closed']} closed"},
    }


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def _template_path(explicit=None):
    for p in ([explicit] if explicit else []) + [
            "dashboard_template.html",
            os.path.join(os.path.dirname(__file__), "..", "dashboard_template.html")]:
        if p and os.path.exists(p):
            return p
    return None


def _render(data_obj, template_path, out_path):
    """Replace the DATA object between the template's DATA_START / DATA_END
    comment markers. The DATA_START marker may carry trailing text inside the
    comment (e.g. '/* DATA_START — ... */'), so we key off the comment, not an
    exact string."""
    with open(template_path) as f:
        tmpl = f.read()
    block = "const DATA = " + json.dumps(data_obj, ensure_ascii=False, indent=2) + ";"
    s = tmpl.find("/* DATA_START")
    e = tmpl.find("/* DATA_END")
    if s == -1 or e == -1:
        raise ValueError("template markers not found")
    s_end = tmpl.find("*/", s)                 # end of the DATA_START comment itself
    if s_end == -1 or s_end > e:
        raise ValueError("template markers not found")
    s_end += 2
    new = tmpl[:s_end] + "\n" + block + "\n" + tmpl[e:]
    with open(out_path, "w") as f:
        f.write(new)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def build(data_dir, template=None, model=None, api_key=None):
    api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
    model = model or os.environ.get("CRAGENT_LLM_MODEL", DEFAULT_MODEL)

    cand_js = _load(os.path.join(data_dir, "candidates_new.json"))
    conf = _load(os.path.join(data_dir, "confirmations.json"))
    led = _load(os.path.join(data_dir, "ledger_status.json"))

    new = cand_js.get("candidates", []) or []
    scan_date = (cand_js.get("scan_date") or conf.get("scan_date")
                 or (led.get("model", {}) or {}).get("scan_date") or "")
    confirmed = conf.get("confirmed_today", []) or []
    pending = conf.get("pending", []) or []

    # the names worth a full write-up: today's new drops + today's confirmed buys
    # (a confirmed buy may have dropped days ago, so it might not be in `new`)
    by_key = {}
    for c in new:
        by_key[(c.get("ticker"), c.get("event_date"))] = c
    for cb in confirmed:
        k = (cb.get("ticker"), cb.get("drop_date"))
        by_key.setdefault(k, {"ticker": cb.get("ticker"), "name": cb.get("name"),
                              "sector": cb.get("sector"), "event_date": cb.get("drop_date")})

    llm_used = False
    cards, narratives = [], {}
    for (tk, ev), cand in by_key.items():
        ann = _announcement(data_dir, tk, ev)
        llm = None
        if api_key:
            llm = _verify_one(cand, ann, api_key, model)
            if llm:
                llm_used = True
        card, narr = _card(cand, ann, llm)
        cards.append(card)
        narratives[tk] = narr

    ledger, sells_due_n = _ledger(led)
    data_obj = {
        "scan_date": _nice_full(scan_date) if scan_date else "today",
        "summary": _summary(new, confirmed, pending, ledger, sells_due_n),
        "confirmed_today": [{"tk": c.get("ticker"), "mk": "US",
                             "sector": c.get("sector"),
                             "entry": c.get("entry_price"),
                             "drop_date": _nice(c.get("drop_date"))} for c in confirmed],
        "pending": [{"tk": p.get("ticker"), "sector": p.get("sector"),
                     "day": p.get("day", "?"), "drop_date": _nice(p.get("drop_date"))}
                    for p in pending],
        "candidates": cards,
        "excluded": "",
        "ledger": ledger,
    }

    # -- write dashboard (fail-soft: a render failure must not kill the email) --
    dash_name = "daily_dashboard.html"
    dash_path = os.path.join(data_dir, dash_name)
    tpath = _template_path(template)
    if tpath:
        try:
            _render(data_obj, tpath, dash_path)
            log(f"dashboard written -> {dash_path} ({len(cards)} card(s), llm={llm_used})")
        except Exception as e:  # noqa: BLE001
            log(f"dashboard render failed: {e!r}")
            dash_name = ""
    else:
        log("dashboard_template.html not found -- skipping dashboard")
        dash_name = ""

    verification = {
        "scan_date": scan_date,
        "nice_date": _nice_full(scan_date) if scan_date else "",
        "llm_used": llm_used,
        "model": model if llm_used else None,
        "dashboard_file": dash_name,
        "by_ticker": narratives,
    }
    with open(os.path.join(data_dir, "verification.json"), "w") as f:
        json.dump(verification, f, indent=2)
    return verification


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--template", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    try:
        v = build(args.data_dir, template=args.template, model=args.model)
        log(f"done: llm_used={v['llm_used']} dashboard={v['dashboard_file'] or '(none)'}")
    except Exception as e:  # noqa: BLE001
        # last-resort guard: never raise into the workflow
        log(f"verify step failed, continuing without enrichment: {e!r}")


if __name__ == "__main__":
    main()
