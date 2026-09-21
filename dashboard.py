"""
dashboard.py
Proctor dashboard: review anonymized snapshot + reason, Confirm/Dismiss/Delete.
"""
import streamlit as st
import json
import os
import glob
from datetime import datetime, timedelta

EVENTS_DIR = "data/events"
st.set_page_config(page_title="Exam Integrity Monitor - Proctor Dashboard", layout="wide")
st.title("Exam Integrity Monitor — Proctor Dashboard")


def load_events():
    events = []
    for json_path in glob.glob(os.path.join(EVENTS_DIR, "*.json")):
        with open(json_path) as f:
            record = json.load(f)
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
    os.remove(record["_json_path"])
    image_path = os.path.join(EVENTS_DIR, record["snapshot"])
    if os.path.exists(image_path):
        os.remove(image_path)


events = load_events()

st.sidebar.header("Filters")
rule_options = sorted(set(e["rule_id"] for e in events)) or ["(none logged yet)"]
selected_rules = st.sidebar.multiselect("Rule type", rule_options, default=rule_options)
status_filter = st.sidebar.selectbox("Reviewer status", ["all", "pending", "confirmed", "dismissed"])

st.sidebar.header("Retention control")
retention_days = st.sidebar.number_input("Delete events older than (days)", min_value=1, value=30)
if st.sidebar.button("Apply retention policy now"):
    cutoff = datetime.now() - timedelta(days=retention_days)
    deleted = 0
    for e in events:
        if datetime.fromtimestamp(e["triggered_at"]) < cutoff:
            delete_event(e)
            deleted += 1
    st.sidebar.success(f"Deleted {deleted} event(s) older than {retention_days} days.")
    st.rerun()

filtered = [e for e in events if e["rule_id"] in selected_rules]
if status_filter != "all":
    filtered = [e for e in filtered if e["reviewer_status"] == status_filter]

st.write(f"Showing {len(filtered)} of {len(events)} total logged events.")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total events", len(events))
col2.metric("Pending review", sum(1 for e in events if e["reviewer_status"] == "pending"))
col3.metric("Confirmed", sum(1 for e in events if e["reviewer_status"] == "confirmed"))
col4.metric("Dismissed", sum(1 for e in events if e["reviewer_status"] == "dismissed"))

st.divider()
for record in filtered:
    with st.container(border=True):
        img_col, info_col = st.columns([1, 2])
        with img_col:
            image_path = os.path.join(EVENTS_DIR, record["snapshot"])
            if os.path.exists(image_path):
                st.image(image_path, caption="Anonymized evidence snapshot")
            else:
                st.warning("Snapshot missing")
        with info_col:
            timestamp = datetime.fromtimestamp(record["triggered_at"]).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"**Rule:** `{record['rule_id']}`")
            st.markdown(f"**When:** {timestamp}")
            st.markdown(f"**Confidence:** {record['confidence']:.2f}")
            st.markdown(f"**Why this fired:** {record['description']}")
            st.markdown(f"**Status:** `{record['reviewer_status']}`")
            b1, b2, b3 = st.columns(3)
            if b1.button("Confirm", key=f"confirm_{record['_json_path']}"):
                save_decision(record, "confirmed")
                st.rerun()
            if b2.button("Dismiss", key=f"dismiss_{record['_json_path']}"):
                save_decision(record, "dismissed")
                st.rerun()
            if b3.button("Delete", key=f"delete_{record['_json_path']}"):
                delete_event(record)
                st.rerun()

if not filtered:
    st.info("No events match the current filters. Run `python run_pipeline.py` to generate some.")
