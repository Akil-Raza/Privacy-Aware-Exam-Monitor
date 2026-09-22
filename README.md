# Privacy-Aware Exam Integrity Monitor

**Project code:** BAI-06 — T.Y. B.Sc. Artificial Intelligence, Semester V
**Course:** KES Shroff College, AY 2026-27

A locally-run exam proctoring prototype that detects integrity-risk behaviors
(gaze/head deviation, multiple people, prohibited objects, prolonged absence)
while anonymizing every stored frame before it ever reaches disk, a log file,
or a reviewer's screen.

## Why privacy-first, not privacy-added-later

Nothing downstream of the anonymization step ever receives the raw camera
frame — not the rule engine, not the event logger, not the dashboard. That
boundary is enforced by what gets *passed between functions* in
`run_pipeline.py`, not just described in a diagram. See "Architecture"
below for exactly where that boundary sits.

## Features

- **Face-presence counting** and **multi-person detection** (MediaPipe Face Mesh)
- **Head-pose (yaw/pitch/roll)** via 6-point `solvePnP`, with a correction
  for a known pitch-ambiguity bug in that exact technique (see Limitations)
- **Iris-based gaze-zone tracking** (left/right from iris position; up/down
  from a per-session-calibrated head-pitch baseline, since iris vertical
  movement is unreliable due to eyelid occlusion — see Limitations)
- **Prohibited-object detection** (phone, laptop, book) via YOLOv8-nano,
  with label-grouping so a detector flickering between similar labels for
  one physical object doesn't inflate the event count
- **Fail-closed anonymization** — if face tracking is lost, the system
  blurs the *entire* frame rather than risk showing raw, identifiable video
- **Temporal rule engine** — every rule requires a *sustained* condition
  (not a single frame) before firing, to suppress false positives
- **Event-level explainability** — every logged event includes a
  deterministic, human-readable reason (`explainability.py`), not just a
  confidence score
- **Proctor dashboard** (Streamlit) — Confirm/Dismiss/Delete per event,
  plus a manual retention policy control
- **Config-driven thresholds** (`config.yaml`) — no hardcoded values buried
  in code
- **Performance instrumentation** — per-stage latency (perception/detection/
  anonymization) and rolling FPS, exported to CSV
- **Evaluation script** (`evaluate.py`) — turns dashboard review decisions
  into false-positive-rate and false-alerts-per-hour metrics

## Architecture

Webcam
-> capture.py (raw frames, throttled to target FPS)
-> perception.py (RAW frame in -> face count, head pose, gaze zone OUT)
-> detection.py (RAW frame in -> object labels/boxes OUT)
-> privacy.py (RAW frame in -> ANONYMIZED frame OUT) <-- privacy boundary
-> rules.py (numbers only: yaw, gaze zone, face count, labels)
-> logger.py (ANONYMIZED frame + event JSON -> data/events/)
-> dashboard.py (reads data/events/ only — separate process)


`perception.py` and `detection.py` both need the raw frame (they can't
detect pixels they can't see) — but their *output* is just numbers and
labels. `rules.py`, `logger.py`, and everything after never receive a raw
frame, only what `privacy.py` already anonymized.

## Setup

This project requires **two separate virtual environments**. `mediapipe`
requires `protobuf<5`; `streamlit` requires `protobuf>=5.26.1`. These
ranges cannot overlap in one environment — this isn't a version-pinning
mistake, it's confirmed unresolvable, so the dashboard and the main
pipeline are deliberately isolated processes that only communicate through
`data/events/` on disk.

```bash
# Main pipeline environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt

# Dashboard environment (separate)
python -m venv venv_dashboard
# Windows:
.\venv_dashboard\Scripts\Activate.ps1
# macOS/Linux:
source venv_dashboard/bin/activate
pip install streamlit
```

## Usage

**Terminal 1 (main venv) — run the monitoring pipeline:**
```bash
python run_pipeline.py
```
Opens a live preview window. Detected events are written to `data/events/`
as JSON + an anonymized snapshot. Press `q` to stop; a performance summary
is written to `data/metrics/performance_summary.csv` on exit.

**Terminal 2 (dashboard venv) — review logged events:**
```bash
streamlit run dashboard.py
```
Opens in your browser. Confirm/Dismiss each event based on what actually
happened, so the numbers below are meaningful.

**After reviewing, generate an evaluation report:**
```bash
python evaluate.py --hours 0.5
```
(`--hours` = duration of the test session being evaluated)

All thresholds (gaze duration, confidence cutoffs, cooldowns, etc.) are in
`config.yaml` — no code changes needed to retune the system.

## Project structure

.
|-- config.yaml # all tunable thresholds
|-- requirements.txt
|-- run_pipeline.py # main pipeline entrypoint
|-- dashboard.py # Streamlit proctor review UI (separate venv)
|-- evaluate.py # false-positive / reviewer-agreement report
|-- src/
| |-- capture.py # Step 2a: video capture
| |-- perception.py # Step 2b: face landmarks, head pose, gaze zone
| |-- privacy.py # Step 2c: fail-closed anonymization
| |-- detection.py # Step 2d: YOLOv8-nano object detection
| |-- rules.py # Phase 3: temporal rule engine
| |-- logger.py # Phase 3: event + snapshot logging
| |-- explainability.py # proctor-readable explanation text
| -- config.py # config.yaml loader -- data/
|-- events/ # JSON + anonymized snapshots (gitignored)
`-- metrics/ # performance CSVs (gitignored)


## Known limitations (found and documented during testing)

- **Head-pose pitch ambiguity:** the 6-point `solvePnP` model is
  near-planar, which has a documented +/-180 degree pose ambiguity.
  Corrected in `perception.py` by snapping out-of-range pitch back into
  the plausible +/-90 degree range every frame (stateless, so it can't drift).
- **Iris vertical gaze is unreliable:** confirmed via testing that
  genuine "looking down" produces iris `dy` values indistinguishable from
  resting noise, due to eyelid occlusion. Up/down gaze uses head pitch
  (relative to a per-session calibrated baseline) instead — iris `dx` is
  used for left/right, where it proved reliable.
- **Small-detector class confusion:** YOLOv8-nano occasionally
  misclassifies a phone as "laptop" (visually similar rectangular
  electronics). Addressed by grouping both labels into one semantic
  category (`device`) in the rule engine, rather than relying on perfect
  classification.
- **Occasional hand/phone false positive:** an empty hand has, in testing,
  been misclassified as a cell phone at >70% confidence. Not fully solved
  in this build; the correct long-term fix is fine-tuning YOLOv8n on
  staged images including empty-hand negatives (a natural baseline-vs-fine-tuned
  comparison experiment, not yet performed).
- **Duplicate-event inflation (fixed):** both `ProhibitedObjectRule` and
  cross-label flicker were found, during testing, to log multiple events
  for one continuous real occurrence. Fixed with a cooldown requiring
  genuine sustained absence before re-firing (see commit history).

## Acknowledgment

An earlier prototype of this project exists in this repository's history
under a different structure. That version had several unresolved issues —
a fail-open anonymization bug, a test that silently never ran due to
incorrect indentation, and frame-rate-dependent rule timing — which were
identified during a review and avoided in this rebuild.
