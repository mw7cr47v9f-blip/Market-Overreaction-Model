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
    model = led.get("model", {}) if isinstance(led, dict) else {}
    holdings = led.get("holdings", []) if isinstance(led, dict) else []

    scan_date = cand.get("scan_date") or model.get("scan_date") or "today"
    new = cand.get("candidates", []) or []
    sells = _sells(model)
    # personal holdings flagged at/near their 3-month mark
    hold_due = [h for h in holdings if str(h.get("flag", "")).upper() in ("SELL_DUE", "SELL_SOON")]
    tally = model.get("realised_tally", {}) or {}

    subj = (f"Cragent.ai daily - {scan_date} - "
            f"{len(new)} new, {len(sells) + len(hold_due)} sell alert(s)")

    # ---- plain text ----
    L = [f"CRAGENT.AI DAILY - {scan_date}", ""]
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
    else:
        L.append("No positions due to sell today.")
        L.append("")

    if new:
        L.append(f"** NEW QUALIFYING NAMES ({len(new)}) **")
        for d in new:
            drop = d.get("drop_pct") or d.get("raw") or d.get("drop")
            L.append(f"  {d.get('ticker')} {d.get('name','')[:30]} ({d.get('sector','')}) "
                     f"drop {_fmt_drop(drop)} - {d.get('trigger_primary','')}")
        L.append("")
    else:
        L.append("No new candidates today.")
        L.append("")

    if tally.get("n_closed"):
        L.append(f"Model book: {model.get('n_open',0)} open, {tally.get('n_closed')} closed, "
                 f"win rate {_fmt_pct((tally.get('win_rate') or 0)*100)}, "
                 f"avg {_fmt_pct((tally.get('avg_return') or 0)*100)}.")
        L.append("")
    L.append("-" * 40)
    L.append(DISCLAIMER)
    text = "\n".join(L)

    # ---- HTML ----
    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    H = [f"<h2 style='font-family:sans-serif;color:#0f766e;margin:0 0 6px'>Cragent.ai daily &middot; {esc(scan_date)}</h2>"]
    if sells or hold_due:
        H.append("<h3 style='font-family:sans-serif;color:#b5482f;margin:12px 0 4px'>Sells due (3-month hold reached)</h3><ul style='font-family:sans-serif;font-size:14px'>")
        for urg, r in sells:
            H.append(f"<li><b>{urg}</b> — {esc(r.get('ticker'))} ({esc(r.get('sector',''))}), entered {esc(r.get('entry_date'))} → {esc(_fmt_pct(r.get('ret_pct')))}, due {esc(r.get('sell_due_date'))}</li>")
        for h in hold_due:
            H.append(f"<li><b>{esc(h.get('flag'))}</b> — {esc(h.get('ticker'))} (your holding) → {esc(_fmt_pct(h.get('ret_pct')))}, due {esc(h.get('sell_due_date'))}</li>")
        H.append("</ul>")
    else:
        H.append("<p style='font-family:sans-serif;font-size:14px'>No positions due to sell today.</p>")
    if new:
        H.append(f"<h3 style='font-family:sans-serif;margin:12px 0 4px'>New qualifying names ({len(new)})</h3><ul style='font-family:sans-serif;font-size:14px'>")
        for d in new:
            drop = d.get("drop_pct") or d.get("raw") or d.get("drop")
            H.append(f"<li><b>{esc(d.get('ticker'))}</b> {esc(str(d.get('name',''))[:30])} ({esc(d.get('sector',''))}) — drop {esc(_fmt_drop(drop))} — {esc(d.get('trigger_primary',''))}</li>")
        H.append("</ul>")
    else:
        H.append("<p style='font-family:sans-serif;font-size:14px'>No new candidates today.</p>")
    if tally.get("n_closed"):
        H.append(f"<p style='font-family:sans-serif;font-size:13px;color:#5b6470'>Model book: {model.get('n_open',0)} open, {tally.get('n_closed')} closed, win rate {esc(_fmt_pct((tally.get('win_rate') or 0)*100))}, avg {esc(_fmt_pct((tally.get('avg_return') or 0)*100))}.</p>")
    H.append(f"<hr><p style='font-family:sans-serif;font-size:10px;color:#8a929c'>{esc(DISCLAIMER)}</p>")
    html = "".join(H)
    return subj, text, html


def send_resend(subj, text, html):
    key = os.environ.get("RESEND_API_KEY")
    to = os.environ.get("CRAGENT_EMAIL_TO")
    frm = os.environ.get("CRAGENT_EMAIL_FROM", "Cragent.ai <onboarding@resend.dev>")
    if not key or not to:
        print("[email] RESEND_API_KEY or CRAGENT_EMAIL_TO not set - skipping send", file=sys.stderr)
        return False
    payload = json.dumps({
        "from": frm,
        "to": [t.strip() for t in to.split(",") if t.strip()],
        "subject": subj,
        "text": text,
        "html": html,
    }).encode()
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
        subj, text, html = build_email(args.data_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[email] build failed: {e!r}", file=sys.stderr)
        return
    if args.print:
        print(subj); print(); print(text)
        return
    send_resend(subj, text, html)


if __name__ == "__main__":
    main()
