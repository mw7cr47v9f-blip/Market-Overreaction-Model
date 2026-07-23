"""
Daily email for the Cragent.ai (Market Overreaction) screen.

Runs after the daily screen, EVERY US trading day — even when there are zero new
candidates — because it carries the 3-month SELL reminders, which are the whole
reason it must land every day. Reads two files the screen just wrote:

    data/candidates_new.json   (the day's new favoured + quality + trigger-gated names)
    data/ledger_status.json    (open book, sells due / due-soon, personal holdings)

and sends one compact HTML+text email via a provider HTTP API (Resend by default;
any provider works via SMTP fallback). Credentials come from env / GitHub secrets:

    RESEND_API_KEY        provider API key (GitHub secret)
    CRAGENT_EMAIL_TO      recipient (comma-separated allowed)
    CRAGENT_EMAIL_FROM    verified sender, e.g. "Cragent.ai <alerts@yourdomain>"
                          (Resend test sender "onboarding@resend.dev" also works)

Never raises into the workflow: on any error it logs and exits 0, so a mail
outage cannot break the screen commit. Not financial advice — see DISCLAIMER.
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request
import urllib.error

DISCLAIMER = (
    "Disclaimer - This is an automated stock screen produced for general information, "
    "educational and entertainment purposes only. It is NOT financial product advice, "
    "investment advice, or a recommendation, offer or solicitation to buy, sell or hold "
    "any security, and it does not take into account your objectives, financial situation "
    "or needs. Nothing in it should be relied upon or acted upon in any way. Screened data "
    "and any figures may be inaccurate, incomplete or out of date, and past performance and "
    "backtested results are not a guide to future performance. Before making any investment "
    "decision, obtain your own independent professional financial and taxation advice and "
    "consider the relevant disclosure documents. To the maximum extent permitted by law, the "
    "author excludes all liability for any loss or damage (including indirect or consequential "
    "loss) arising from any use of, or reliance on, this material. By reading it you accept "
    "these terms."
)


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _fmt_pct(x):
    try:
        return f"{float(x):+.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_drop(x):
    """Drop fields are stored as fractions (-0.21 = -21%); scale to percent."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    if abs(v) <= 1.5:          # fraction -> percent
        v *= 100.0
    return f"{v:+.0f}%"


def _narr_text(ticker, narr):
    """Plain-text verification lines for one ticker (empty if none)."""
    n = (narr or {}).get(ticker) or {}
    out = []
    if n.get("read"):
        out.append(f"      {n['read']}")
    bits = [b for b in (n.get("trigger"), n.get("detail")) if b]
    if bits:
        out.append("      " + " - ".join(bits))
    if n.get("insider"):
        out.append(f"      Insider: {n['insider']}")
    return out


def _narr_html(ticker, narr, ff, esc):
    """Verification HTML for one ticker (empty string if none)."""
    n = (narr or {}).get(ticker) or {}
    if not n:
        return ""
    parts = []
    if n.get("read"):
        parts.append(f"<div style='{ff};font-size:13px;color:#38485a;margin:3px 0 0'>{esc(n['read'])}</div>")
    sub = [esc(b) for b in (n.get("trigger"), n.get("detail")) if b]
    if sub:
        parts.append(f"<div style='{ff};font-size:12px;color:#6b7a8d;margin:2px 0 0'>{' &mdash; '.join(sub)}</div>")
    if n.get("insider"):
        parts.append(f"<div style='{ff};font-size:12px;color:#6b7a8d;margin:1px 0 0'>Insider: {esc(n['insider'])}</div>")
    return "".join(parts)


def _sells(model):
    """Positions to sell NOW (hit 3-month mark this run) and SOON (within a week),
    plus any personal holdings flagged. Returns a list of (urgency, dict)."""
    out = []
    for r in model.get("closed_this_run", []) or []:
        out.append(("SELL NOW", r))
    for r in model.get("sell_soon", []) or []:
        out.append(("SELL SOON", r))
    return out


def build_email(data_dir):
    cand = _load(os.path.join(data_dir, "candidates_new.json"))
    led = _load(os.path.join(data_dir, "ledger_status.json"))
    conf = _load(os.path.join(data_dir, "confirmations.json"))
    verif = _load(os.path.join(data_dir, "verification.json"))
    narr = verif.get("by_ticker", {}) if isinstance(verif, dict) else {}
    llm_used = bool(verif.get("llm_used")) if isinstance(verif, dict) else False
    dash_file = verif.get("dashboard_file") if isinstance(verif, dict) else ""
    dash_path = os.path.join(data_dir, dash_file) if dash_file else ""
    model = led.get("model", {}) if isinstance(led, dict) else {}
    holdings = led.get("holdings", []) if isinstance(led, dict) else []

    scan_date = conf.get("scan_date") or cand.get("scan_date") or model.get("scan_date") or "today"
    new_drops = cand.get("candidates", []) or []
    confirmed = conf.get("confirmed_today", []) or []      # BUY today (breakout fired)
    pending = conf.get("pending", []) or []                # awaiting confirmation (day N/15)
    sells = _sells(model)
    hold_due = [h for h in holdings if str(h.get("flag", "")).upper() in ("SELL_DUE", "SELL_SOON")]
    tally = model.get("realised_tally", {}) or {}

    # ---- session-date label + staleness guard ------------------------------
    # scan_date is the newest close actually in the data. Label the report by that
    # trading session (not a calendar "today"), and flag if the feed looks stale.
    from datetime import date as _date, datetime as _dt, timezone as _tz
    _MONTHS = ["January","February","March","April","May","June","July","August",
               "September","October","November","December"]
    try:
        _sd = _date.fromisoformat(str(scan_date)[:10])
        nice_date = f"{_sd.day} {_MONTHS[_sd.month-1]} {_sd.year}"
        _age = (_dt.now(_tz.utc).date() - _sd).days
        stale = _age > 3        # more than a long weekend behind => probably a stale feed
    except Exception:
        nice_date, _age, stale = str(scan_date), None, False
    gen_utc = _dt.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC")
    stale_txt = (f"DATA MAY BE STALE - the newest close in the feed is {nice_date} "
                 f"({_age} days old). The price feed may not have updated; treat today's "
                 f"signals with caution.") if stale else ""

    subj = (f"Cragent.ai - End of Trading Day, {nice_date} - "
            f"{len(confirmed)} BUY, {len(sells) + len(hold_due)} sell, {len(new_drops)} new drop(s)")

    # ---- plain text ----
    L = [f"CRAGENT.AI - END OF TRADING DAY, {nice_date}", f"(report generated {gen_utc})", ""]
    if stale_txt:
        L = [f"!! {stale_txt}", ""] + L
    if confirmed:
        L.append("** BUY NOW - confirmed entry today (two-day-high breakout on volume) **")
        for c in confirmed:
            L.append(f"  {c.get('ticker')} ({c.get('sector','')}) confirmed @ {c.get('entry_price')} "
                     f"- dropped {c.get('drop_date')}")
            L.extend(_narr_text(c.get("ticker"), narr))
        L.append("")
    if sells or hold_due:
        L.append("** SELLS DUE (3-month hold reached) **")
        for urg, r in sells:
            L.append(f"  [{urg}] {r.get('ticker')} ({r.get('sector','')}) "
                     f"entered {r.get('entry_date')} -> {_fmt_pct(r.get('ret_pct'))}, "
                     f"due {r.get('sell_due_date')}")
        for h in hold_due:
            L.append(f"  [{h.get('flag')}] {h.get('ticker')} (your holding) "
                     f"-> {_fmt_pct(h.get('ret_pct'))}, due {h.get('sell_due_date')}")
        L.append("")
    if not confirmed and not (sells or hold_due):
        L.append("No buys or sells today.")
        L.append("")
    if pending:
        L.append(f"** AWAITING CONFIRMATION ({len(pending)}) - not yet bought **")
        for p in pending:
            L.append(f"  {p.get('ticker')} ({p.get('sector','')}) day {p.get('day','?')}/15 "
                     f"since drop {p.get('drop_date')}")
            L.extend(_narr_text(p.get("ticker"), narr))
        L.append("")

    if tally.get("n_closed"):
        L.append(f"Model book: {model.get('n_open',0)} open, {tally.get('n_closed')} closed, "
                 f"win rate {_fmt_pct((tally.get('win_rate') or 0)*100)}, "
                 f"avg {_fmt_pct((tally.get('avg_return') or 0)*100)}.")
        L.append("")
    ver_note = (f"Verification: primary-source (SEC EDGAR) filing review by Claude "
                f"({verif.get('model') or 'model'})." if llm_used
                else "Verification: mechanical screen only this run (no model review).")
    L.append(ver_note)
    if dash_path and os.path.exists(dash_path):
        L.append("Full dashboard attached (open the .html for the ledger + per-name analysis).")
    L.append("")
    L.append("-" * 40)
    L.append(DISCLAIMER)
    text = "\n".join(L)

    # ---- HTML ----
    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    ff = "font-family:sans-serif"
    H = []
    if stale_txt:
        H.append(f"<div style='{ff};background:#fdecea;border:1px solid #f5c2bd;color:#a8331f;"
                 f"padding:10px 12px;border-radius:8px;margin:0 0 10px;font-size:13px'>"
                 f"&#9888; {esc(stale_txt)}</div>")
    H.append(f"<h2 style='{ff};color:#0f766e;margin:0 0 2px'>Cragent.ai &middot; End of Trading Day, {esc(nice_date)}</h2>")
    H.append(f"<div style='{ff};color:#5b6470;font-size:12px;margin:0 0 8px'>report generated {esc(gen_utc)}</div>")
    if confirmed:
        H.append(f"<h3 style='{ff};color:#0f766e;margin:12px 0 4px'>Buy now &mdash; confirmed entry today</h3><ul style='{ff};font-size:14px'>")
        for c in confirmed:
            H.append(f"<li><b>{esc(c.get('ticker'))}</b> ({esc(c.get('sector',''))}) &mdash; confirmed breakout @ {esc(c.get('entry_price'))}, dropped {esc(c.get('drop_date'))}"
                     f"{_narr_html(c.get('ticker'), narr, ff, esc)}</li>")
        H.append("</ul>")
    if sells or hold_due:
        H.append(f"<h3 style='{ff};color:#b5482f;margin:12px 0 4px'>Sells due (3-month hold reached)</h3><ul style='{ff};font-size:14px'>")
        for urg, r in sells:
            H.append(f"<li><b>{urg}</b> — {esc(r.get('ticker'))} ({esc(r.get('sector',''))}), entered {esc(r.get('entry_date'))} → {esc(_fmt_pct(r.get('ret_pct')))}, due {esc(r.get('sell_due_date'))}</li>")
        for h in hold_due:
            H.append(f"<li><b>{esc(h.get('flag'))}</b> — {esc(h.get('ticker'))} (your holding) → {esc(_fmt_pct(h.get('ret_pct')))}, due {esc(h.get('sell_due_date'))}</li>")
        H.append("</ul>")
    if not confirmed and not (sells or hold_due):
        H.append(f"<p style='{ff};font-size:14px'>No buys or sells today.</p>")
    if pending:
        H.append(f"<h3 style='{ff};margin:12px 0 4px'>Awaiting confirmation ({len(pending)}) &mdash; not yet bought</h3><ul style='{ff};font-size:13px;color:#5b6470'>")
        for p in pending:
            H.append(f"<li>{esc(p.get('ticker'))} ({esc(p.get('sector',''))}) — day {esc(p.get('day','?'))}/15 since drop {esc(p.get('drop_date'))}"
                     f"{_narr_html(p.get('ticker'), narr, ff, esc)}</li>")
        H.append("</ul>")
    if tally.get("n_closed"):
        H.append(f"<p style='{ff};font-size:13px;color:#5b6470'>Model book: {model.get('n_open',0)} open, {tally.get('n_closed')} closed, win rate {esc(_fmt_pct((tally.get('win_rate') or 0)*100))}, avg {esc(_fmt_pct((tally.get('avg_return') or 0)*100))}.</p>")
    if llm_used:
        H.append(f"<p style='{ff};font-size:11px;color:#8a929c;margin:10px 0 0'>Verification: primary-source (SEC EDGAR) filing review by Claude ({esc(verif.get('model') or 'model')}).</p>")
    else:
        H.append(f"<p style='{ff};font-size:11px;color:#8a929c;margin:10px 0 0'>Verification: mechanical screen only this run (no model review).</p>")
    if dash_path and os.path.exists(dash_path):
        H.append(f"<p style='{ff};font-size:11px;color:#8a929c;margin:2px 0 0'>Full dashboard attached &mdash; open the .html for the ledger and per-name analysis.</p>")
    H.append(f"<hr><p style='{ff};font-size:10px;color:#8a929c'>{esc(DISCLAIMER)}</p>")
    html = "".join(H)

    # ---- dashboard attachment (base64 for the provider API) ----
    attachments = []
    if dash_path and os.path.exists(dash_path):
        try:
            import base64
            with open(dash_path, "rb") as f:
                attachments.append({
                    "filename": f"{nice_date} Cragent.ai Daily Report.html",
                    "content": base64.b64encode(f.read()).decode(),
                })
        except Exception as e:  # noqa: BLE001
            print(f"[email] could not attach dashboard: {e!r}", file=sys.stderr)
    return subj, text, html, attachments


def send_resend(subj, text, html, attachments=None):
    key = os.environ.get("RESEND_API_KEY")
    to = os.environ.get("CRAGENT_EMAIL_TO")
    frm = os.environ.get("CRAGENT_EMAIL_FROM", "Cragent.ai <onboarding@resend.dev>")
    if not key or not to:
        print("[email] RESEND_API_KEY or CRAGENT_EMAIL_TO not set - skipping send", file=sys.stderr)
        return False
    msg = {
        "from": frm,
        "to": [t.strip() for t in to.split(",") if t.strip()],
        "subject": subj,
        "text": text,
        "html": html,
    }
    if attachments:
        msg["attachments"] = attachments
    payload = json.dumps(msg).encode()
    req = urllib.request.Request(
        "https://api.resend.com/emails", data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"[email] sent ({resp.status})", file=sys.stderr)
            return True
    except urllib.error.HTTPError as e:
        print(f"[email] provider error {e.code}: {e.read()[:200]!r}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[email] send failed: {e!r}", file=sys.stderr)
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--print", action="store_true", help="print the email instead of sending")
    args = ap.parse_args()
    try:
        subj, text, html, attachments = build_email(args.data_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[email] build failed: {e!r}", file=sys.stderr)
        return
    if args.print:
        print(subj); print(); print(text)
        print(f"\n[attachments: {len(attachments)}]")
        return
    send_resend(subj, text, html, attachments)


if __name__ == "__main__":
    main()
