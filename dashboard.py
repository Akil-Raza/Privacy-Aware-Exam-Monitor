"""
dashboard.py
Proctor dashboard: review each anonymised snapshot together with the reason
it was flagged, then confirm, dismiss or delete it.

This process only reads data/events/. It never touches the camera or any raw
frame, which is why it can run in a separate environment from the pipeline.
"""
import base64
import csv
import glob
import html
import io
import json
import os
from datetime import datetime, timedelta

import streamlit as st

EVENTS_DIR = "data/events"
PAGE_SIZE = 12

# rule id -> (display name, colour)
RULES = {
    "GAZE_AWAY": ("Gaze away", "#0ea5e9"),
    "MULTIPLE_FACES": ("Multiple faces", "#a855f7"),
    "PROLONGED_ABSENCE": ("Prolonged absence", "#14b8a6"),
    "PROHIBITED_OBJECT": ("Prohibited object", "#ef4444"),
}
# review status -> (display name, colour)
STATUS = {
    "pending": ("Pending", "#f59e0b"),
    "confirmed": ("Confirmed", "#22c55e"),
    "dismissed": ("Dismissed", "#94a3b8"),
}

CSS = """
<style>
.block-container { padding-top: 3.6rem; max-width: 1500px; }
[data-testid="stAppDeployButton"], #MainMenu { display: none; }

.hdr { display: flex; align-items: flex-end; justify-content: space-between;
       flex-wrap: wrap; gap: .6rem; margin-bottom: 1.1rem; }
.hdr h1 { font-size: 1.75rem; font-weight: 700; margin: 0; padding: 0;
          letter-spacing: -0.01em; line-height: 1.2; }
.hdr .sub { opacity: .65; font-size: .95rem; margin-top: .2rem; }
.pill { display: inline-block; border: 1px solid rgba(34,197,94,.55); color: #22c55e;
        background: rgba(34,197,94,.10); border-radius: 999px; padding: .25rem .8rem;
        font-size: .8rem; font-weight: 600; }
.updated { text-align: right; opacity: .55; font-size: .8rem; margin-bottom: .4rem; }

.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: .8rem; margin-bottom: 1rem; }
.kpi { border: 1px solid rgba(128,128,128,.25); border-left: 4px solid var(--c);
       border-radius: 12px; padding: .8rem 1rem; background: rgba(128,128,128,.08); }
.kpi .v { font-size: 1.9rem; font-weight: 700; line-height: 1.15; }
.kpi .l { font-size: .74rem; opacity: .7; text-transform: uppercase; letter-spacing: .07em; }
@media (max-width: 1000px) { .kpis { grid-template-columns: repeat(2, 1fr); } }

.mix { border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .8rem 1rem;
       background: rgba(128,128,128,.08); margin-bottom: 1.1rem; }
.mix .t { font-size: .74rem; opacity: .7; text-transform: uppercase; letter-spacing: .07em;
          margin-bottom: .5rem; }
.bar { display: flex; height: 10px; border-radius: 6px; overflow: hidden;
       background: rgba(128,128,128,.2); }
.legend { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; margin-top: .6rem; font-size: .85rem; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .4rem; }

.imgwrap { position: relative; border-radius: 10px; overflow: hidden; background: #000; }
.evt-img { width: 100%; aspect-ratio: 4 / 3; object-fit: contain; display: block; background: #000; }
.evt-img.missing { display: flex; align-items: center; justify-content: center;
                   color: #94a3b8; font-size: .9rem; }
.imgwrap .st { position: absolute; top: 10px; right: 10px; }
.badge { display: inline-block; padding: .16rem .6rem; border-radius: 999px; font-size: .75rem;
         font-weight: 600; color: #fff; }
.row { display: flex; align-items: center; justify-content: space-between; gap: .5rem;
       margin: .7rem 0 .5rem 0; flex-wrap: wrap; }
.when { opacity: .65; font-size: .82rem; }
.confrow { display: flex; justify-content: space-between; font-size: .82rem; margin-bottom: .25rem; }
.confrow span { opacity: .65; }
.conf { height: 6px; border-radius: 4px; background: rgba(128,128,128,.25); overflow: hidden; }
.conf > div { height: 100%; border-radius: 4px; }
.why { border-left: 3px solid var(--c); padding: .5rem .8rem; background: rgba(128,128,128,.08);
       border-radius: 0 8px 8px 0; font-size: .92rem; margin: .8rem 0 .4rem 0; line-height: 1.45;
       min-height: 4.6rem; }
.why b { display: block; font-size: .72rem; opacity: .65; text-transform: uppercase;
         letter-spacing: .07em; margin-bottom: .15rem; }

div[class*="st-key-card_"] { border-radius: 14px; }
div[class*="st-key-card_"] [data-testid="stElementContainer"],
div[class*="st-key-card_"] [data-testid="stButton"],
div[class*="st-key-card_"] [data-testid="stPopover"] { width: 100% !important; }
div[class*="st-key-card_"] [data-testid="stLayoutWrapper"],
div[class*="st-key-card_"] [data-testid="stPopover"] > div { width: 100% !important; }
div[class*="st-key-card_"] [data-testid="stButton"] > button,
div[class*="st-key-card_"] [data-testid="stPopoverButton"] { width: 100% !important; }
button[data-testid="stBaseButton-primary"] { background-color: #16a34a; border-color: #16a34a; color: #fff; }
button[data-testid="stBaseButton-primary"]:hover:not(:disabled) { background-color: #15803d; border-color: #15803d; color: #fff; }
.empty { border: 1px dashed rgba(128,128,128,.45); border-radius: 14px; padding: 2.2rem 1rem;
         text-align: center; opacity: .85; }
.empty code { font-size: .9rem; }
</style>
"""


# ---------- data ----------
def load_events():
    events = []
    for json_path in glob.glob(os.path.join(EVENTS_DIR, "*.json")):
        try:
            with open(json_path) as f:
                record = json.load(f)
            record["triggered_at"] = float(record["triggered_at"])
            _ = record["rule_id"]
        except (json.JSONDecodeError, KeyError, OSError, ValueError):
            continue  # file still being written by the pipeline, or damaged
        record["_json_path"] = json_path
        record.setdefault("reviewer_status", "pending")
        events.append(record)
    events.sort(key=lambda e: e["triggered_at"], reverse=True)
    return events


def save_decision(record, decision):
    record["reviewer_status"] = decision
    clean = {k: v for k, v in record.items() if not k.startswith("_")}
    with open(record["_json_path"], "w") as f:
        json.dump(clean, f, indent=2)


def delete_event(record):
    if os.path.exists(record["_json_path"]):
        os.remove(record["_json_path"])
    image_path = os.path.join(EVENTS_DIR, record.get("snapshot", ""))
    if record.get("snapshot") and os.path.exists(image_path):
        os.remove(image_path)


@st.cache_data(show_spinner=False)
def image_data_uri(path, mtime):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def rule_label(rule_id):
    return RULES.get(rule_id, (rule_id.replace("_", " ").title(), ""))[0]


def time_ago(ts):
    s = max(0, int(datetime.now().timestamp() - ts))
    if s < 60:
        return f"{s} s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    if s < 86400:
        return f"{s // 3600} h ago"
    return f"{s // 86400} d ago"


def events_to_csv(events):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["time", "rule", "confidence", "review_status", "explanation", "snapshot"])
    for e in events:
        writer.writerow([
            datetime.fromtimestamp(e["triggered_at"]).strftime("%Y-%m-%d %H:%M:%S"),
            e["rule_id"], f"{float(e.get('confidence', 0)):.2f}", e["reviewer_status"],
            e.get("description", ""), e.get("snapshot", ""),
        ])
    return buf.getvalue()


# ---------- page ----------
st.set_page_config(page_title="Proctor Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Review controls")
    all_events = load_events()
    known = list(RULES.keys())
    rule_ids = known + sorted(set(e["rule_id"] for e in all_events) - set(known))
    selected_rules = st.multiselect("Rule type", rule_ids, default=rule_ids, format_func=rule_label)
    sort_by = st.selectbox("Sort by", ["Newest first", "Oldest first", "Highest confidence",
                                       "Lowest confidence", "Pending first"])
    live = st.toggle("Live refresh (every 5 s)", value=True,
                     help="Shows new events as the pipeline logs them.")

    st.markdown("### Retention")
    retention_days = st.number_input("Delete events older than (days)", min_value=1, value=30)
    cutoff = datetime.now() - timedelta(days=retention_days)
    old_events = [e for e in all_events if datetime.fromtimestamp(e["triggered_at"]) < cutoff]
    st.caption(f"{len(old_events)} event(s) are older than {retention_days} days.")
    if st.button("Apply retention policy", disabled=not old_events):
        for e in old_events:
            delete_event(e)
        st.session_state["flash"] = f"Deleted {len(old_events)} event(s) older than {retention_days} days."
        st.rerun()
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    st.markdown("### Export")
    st.download_button("Download event log (CSV)", data=events_to_csv(all_events),
                       file_name="event_log.csv", mime="text/csv", disabled=not all_events)
    st.caption("Snapshots are anonymised before they are saved. No raw video is stored.")

st.markdown(
    '<div class="hdr"><div><h1>Exam Integrity Monitor</h1>'
    '<div class="sub">Proctor review dashboard</div></div>'
    '<span class="pill">Anonymised evidence only</span></div>',
    unsafe_allow_html=True,
)


def kpi(value, label, colour):
    return f'<div class="kpi" style="--c:{colour}"><div class="v">{value}</div><div class="l">{label}</div></div>'


def render_card(rec):
    name, colour = RULES.get(rec["rule_id"], (rule_label(rec["rule_id"]), "#64748b"))
    status = rec["reviewer_status"]
    status_name, status_colour = STATUS.get(status, (status.title(), "#64748b"))
    confidence = max(0.0, min(1.0, float(rec.get("confidence", 0))))
    when = datetime.fromtimestamp(rec["triggered_at"])
    key = os.path.splitext(os.path.basename(rec["_json_path"]))[0]

    image_path = os.path.join(EVENTS_DIR, rec.get("snapshot", ""))
    if rec.get("snapshot") and os.path.exists(image_path):
        uri = image_data_uri(image_path, os.path.getmtime(image_path))
        image_html = f'<img class="evt-img" src="{uri}">'
    else:
        image_html = '<div class="evt-img missing">Snapshot missing</div>'

    with st.container(border=True, key=f"card_{key}"):
        st.markdown(
            f'<div class="imgwrap">{image_html}'
            f'<span class="badge st" style="background:{status_colour}">{status_name}</span></div>'
            f'<div class="row"><span class="badge" style="background:{colour}">{html.escape(name)}</span>'
            f'<span class="when">{when:%d %b %Y, %H:%M:%S} &middot; {time_ago(rec["triggered_at"])}</span></div>'
            f'<div class="confrow"><span>Confidence</span><b>{confidence:.0%}</b></div>'
            f'<div class="conf"><div style="width:{confidence * 100:.0f}%;background:{colour}"></div></div>'
            f'<div class="why" style="--c:{colour}"><b>Why this fired</b>{html.escape(rec.get("description", ""))}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Technical details"):
            st.json(rec.get("details", {}))
        b1, b2, b3 = st.columns(3)
        if b1.button("Confirm", key=f"confirm_{key}", type="primary", disabled=status == "confirmed"):
            save_decision(rec, "confirmed")
            st.rerun()
        if b2.button("Dismiss", key=f"dismiss_{key}", disabled=status == "dismissed"):
            save_decision(rec, "dismissed")
            st.rerun()
        with b3.popover("Delete"):
            st.caption("Removes this record and its snapshot.")
            if st.button("Yes, delete", key=f"delete_{key}"):
                delete_event(rec)
                st.rerun()


def board():
    events = load_events()
    total = len(events)
    counts = {s: sum(1 for e in events if e["reviewer_status"] == s) for s in STATUS}
    reviewed = counts["confirmed"] + counts["dismissed"]
    fp_rate = f'{counts["dismissed"] / reviewed:.0%}' if reviewed else "--"

    st.markdown(f'<div class="updated">Last updated {datetime.now():%H:%M:%S}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="kpis">'
        + kpi(total, "Total events", "#64748b")
        + kpi(counts["pending"], "Pending review", STATUS["pending"][1])
        + kpi(counts["confirmed"], "Confirmed", STATUS["confirmed"][1])
        + kpi(counts["dismissed"], "Dismissed", STATUS["dismissed"][1])
        + kpi(fp_rate, "False-positive rate", "#6366f1")
        + "</div>",
        unsafe_allow_html=True,
    )

    if total:
        by_rule = {}
        for e in events:
            by_rule[e["rule_id"]] = by_rule.get(e["rule_id"], 0) + 1
        segments = "".join(
            f'<div style="width:{n / total * 100:.1f}%;background:{RULES.get(r, ("", "#64748b"))[1]}"></div>'
            for r, n in by_rule.items()
        )
        legend = "".join(
            f'<span><span class="dot" style="background:{RULES.get(r, ("", "#64748b"))[1]}"></span>'
            f'{html.escape(rule_label(r))} <b>{n}</b></span>'
            for r, n in sorted(by_rule.items(), key=lambda kv: -kv[1])
        )
        st.markdown(
            f'<div class="mix"><div class="t">Events by rule</div><div class="bar">{segments}</div>'
            f'<div class="legend">{legend}</div></div>',
            unsafe_allow_html=True,
        )

    status_filter = st.radio("Show", ["All", "Pending", "Confirmed", "Dismissed"], horizontal=True,
                             key="status_filter")
    shown = [e for e in events if e["rule_id"] in selected_rules]
    if status_filter != "All":
        shown = [e for e in shown if e["reviewer_status"] == status_filter.lower()]

    if sort_by == "Oldest first":
        shown.sort(key=lambda e: e["triggered_at"])
    elif sort_by == "Highest confidence":
        shown.sort(key=lambda e: -float(e.get("confidence", 0)))
    elif sort_by == "Lowest confidence":
        shown.sort(key=lambda e: float(e.get("confidence", 0)))
    elif sort_by == "Pending first":
        shown.sort(key=lambda e: (e["reviewer_status"] != "pending", -e["triggered_at"]))

    if not total:
        st.markdown(
            '<div class="empty"><b>No events logged yet</b><br>Start the pipeline with '
            '<code>python run_pipeline.py</code> and flagged events will appear here.</div>',
            unsafe_allow_html=True,
        )
        return
    if not shown:
        st.markdown('<div class="empty">No events match the current filters.</div>', unsafe_allow_html=True)
        return

    st.caption(f"Showing {min(len(shown), st.session_state.get('limit', PAGE_SIZE))} of {len(shown)} matching events")
    limit = st.session_state.setdefault("limit", PAGE_SIZE)
    visible = shown[:limit]
    for i in range(0, len(visible), 2):
        cols = st.columns(2, gap="medium")
        for col, rec in zip(cols, visible[i:i + 2]):
            with col:
                render_card(rec)
    if len(shown) > limit:
        if st.button(f"Show {PAGE_SIZE} more"):
            st.session_state["limit"] = limit + PAGE_SIZE
            st.rerun(scope="fragment")


st.fragment(run_every=5 if live else None)(board)()