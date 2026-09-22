"""
evaluate.py
Evaluation report from human-reviewed events: false alerts per hour,
reviewer agreement, breakdown by rule.
"""
import json
import glob
import os
import argparse
from collections import Counter


def load_events(events_dir="data/events"):
    events = []
    for path in glob.glob(os.path.join(events_dir, "*.json")):
        with open(path) as f:
            events.append(json.load(f))
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, required=True)
    args = parser.parse_args()

    events = load_events()
    if not events:
        print("No events found. Run run_pipeline.py first.")
        return

    total = len(events)
    reviewed = [e for e in events if e.get("reviewer_status", "pending") != "pending"]
    confirmed = [e for e in reviewed if e["reviewer_status"] == "confirmed"]
    dismissed = [e for e in reviewed if e["reviewer_status"] == "dismissed"]

    print(f"Total events logged:      {total}")
    print(f"Reviewed:                 {len(reviewed)} ({total - len(reviewed)} still pending in dashboard)")
    print(f"Confirmed (true alerts):  {len(confirmed)}")
    print(f"Dismissed (false alerts): {len(dismissed)}")
    if reviewed:
        fp_rate = len(dismissed) / len(reviewed) * 100
        print(f"\nFalse positive rate:      {fp_rate:.1f}% of reviewed events")
    print(f"False alerts per hour:    {len(dismissed) / args.hours:.2f}")

    print("\nBreakdown by rule (total logged vs. dismissed):")
    by_rule = Counter(e["rule_id"] for e in events)
    dismissed_by_rule = Counter(e["rule_id"] for e in dismissed)
    for rule_id, count in by_rule.items():
        print(f"  {rule_id:20s} total={count:3d}  dismissed={dismissed_by_rule.get(rule_id, 0):3d}")

    avg_conf = sum(e["confidence"] for e in events) / total
    print(f"\nAverage confidence across all events: {avg_conf:.2f}")


if __name__ == "__main__":
    main()
