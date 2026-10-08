# Privacy-Aware Exam Integrity Monitor

Project code BAI-06, T.Y. B.Sc. Artificial Intelligence, Semester V
KES' Shroff College, AY 2026-27

A proctoring prototype that runs entirely on the student's own machine. It watches a webcam feed for behaviour that may point to cheating (looking away, extra people, leaving the seat, phones or books) and blurs every frame before anything is saved or shown to a reviewer. Each flagged event comes with a plain-English reason, so a human invigilator can check it and decide. The system never penalises anyone by itself.

## What it does

| Rule | Fires when | Confidence |
|---|---|---|
| `GAZE_AWAY` | head yaw above 25 degrees, or gaze zone not centre, for 5 s | grows with duration, about 0.5 when it fires |
| `MULTIPLE_FACES` | 2 or more faces in view for 3 s | fixed 0.9 |
| `PROLONGED_ABSENCE` | no face in view for 8 s | fixed 0.85 |
| `PROHIBITED_OBJECT` | phone, laptop or book seen at 0.6 confidence or more | YOLO confidence |

Each rule fires once per episode, not once per frame, so one long phone use gives one event. Confidence scores are heuristic scores, not calibrated probabilities.

## Privacy design

- The raw camera frame only exists in memory, inside the perception, detection and anonymisation steps. The rule engine, logger and dashboard never receive it.
- Faces are blurred before anything is stored or displayed.
- The anonymiser fails closed. If no face can be found for 10 frames (about one second at 10 fps), the whole frame is blurred instead of risking a visible face.
- Only two things are written to disk per event: a small JSON record and a blurred JPG snapshot. No video is ever saved.
- The dashboard has a retention control that deletes events older than a chosen number of days.
- A human reviews every event. The system flags, the invigilator decides.

The design follows the ideas of data minimisation, storage limitation, human oversight of automated decisions and data protection by design found in GDPR (Articles 5, 22 and 25) and India's Digital Personal Data Protection Act, 2023. It has not had a legal review, so it is described as aligned with these, not certified compliant.

## Features

- Face counting and multi-person detection (MediaPipe Face Mesh)
- Head pose (yaw, pitch, roll) from six face landmarks and `solvePnP`, with a fix for the known near-planar pitch ambiguity
- Iris-based gaze zones: left and right from iris position, up and down from head pitch measured against a per-session baseline (the first 30 frames)
- Prohibited object detection (phone, laptop, book) with YOLOv8-nano, running on CPU
- Label grouping, so a detector that flips between "phone" and "laptop" for one object still counts as one event
- Rule engine based on elapsed time, so behaviour is the same at any frame rate
- Event-level explanations from deterministic templates (`explainability.py`), which can be audited line by line
- Streamlit proctor dashboard: filter by rule and review status, confirm, dismiss or delete events, apply a retention policy
- All thresholds in `config.yaml`, with no values hidden in the code
- Performance timing for each stage and a rolling processing FPS, exported to CSV
- `evaluate.py`, which turns the invigilator's decisions into a false-positive rate and false alerts per hour

## How it works

```
Webcam
  -> capture.py       raw frames, limited to the target FPS
  -> perception.py    raw frame in, face count + head pose + gaze zone out
  -> detection.py     raw frame in, object labels + boxes out
  -> privacy.py       raw frame in, anonymised frame out   <-- privacy boundary
  -> rules.py         numbers and labels only
  -> explainability.py  reason text for each event
  -> logger.py        anonymised frame + event JSON -> data/events/
  -> dashboard.py     reads data/events/ only (separate process)
```

`perception.py` and `detection.py` need the raw frame because they cannot find a face or a phone without pixels. What they return is numbers and labels. Everything after `privacy.py` only sees the anonymised frame.

## Setup

The project needs two virtual environments. `mediapipe` needs `protobuf<5` and `streamlit` needs `protobuf>=5.26.1`, and the two ranges cannot be installed together. So the pipeline and the dashboard run as separate processes and only share the `data/events/` folder.

Windows (PowerShell):

```powershell
# Pipeline environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
deactivate

# Dashboard environment
python -m venv venv_dashboard
.\venv_dashboard\Scripts\Activate.ps1
pip install streamlit
deactivate
```

macOS and Linux:

```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && deactivate
python -m venv venv_dashboard && source venv_dashboard/bin/activate && pip install streamlit && deactivate
```

The first run downloads the `yolov8n.pt` weights, so it needs an internet connection once.

## Running it

Windows, one command:

```powershell
.\start_demo.bat
```

This opens two windows: the live camera feed with detection overlays, and the dashboard in your browser.

On macOS and Linux (or if you prefer two terminals):

```bash
# Terminal 1, pipeline environment
python run_pipeline.py

# Terminal 2, dashboard environment
streamlit run dashboard.py
```

Tips for a clean run:

- Close Zoom, Teams or anything else that is using the camera.
- Sit facing the screen for about 3 seconds when it starts. That is the pitch calibration, and up/down gaze is not judged until it finishes.
- Press `q` in the camera window to stop. A performance summary is written to `data/metrics/performance_summary.csv`.

## Reviewing and evaluating

1. Open the dashboard and mark each event as confirmed or dismissed, based on what really happened.
2. Run the evaluation, giving the length of the test session in hours:

```bash
python evaluate.py --hours 0.5
```

It reports the number of events per rule, the share confirmed and dismissed, the false-positive rate and the false alerts per hour.

## Event format

Each event is saved as `data/events/YYYYMMDD_HHMMSS_RULEID.json` with a matching `.jpg` snapshot of the same name. Example (values are illustrative):

```json
{
  "rule_id": "PROHIBITED_OBJECT",
  "description": "A cell phone was detected in frame. Confidence: 78%.",
  "confidence": 0.78,
  "triggered_at": "2026-10-05T10:15:30",
  "details": { "label": "cell phone" },
  "snapshot": "20261005_101530_PROHIBITED_OBJECT.jpg"
}
```

The dashboard adds a `reviewer_status` field when an event is confirmed or dismissed.

## Configuration

Everything tunable is in `config.yaml`:

| Setting | Value | Meaning |
|---|---|---|
| `capture.target_fps` | 10 | frames processed per second |
| `privacy.grace_frames` | 10 | frames without a face before the whole frame is blurred |
| `detection.confidence_threshold` | 0.5 | general YOLO threshold |
| `detection.class_overrides` | phone 0.30, laptop 0.30 | lower threshold for small or partly hidden objects |
| `rules.gaze_yaw_threshold` | 25.0 | head turn in degrees |
| `rules.gaze_duration_sec` | 5.0 | how long before gaze-away fires |
| `rules.multi_face_duration_sec` | 3.0 | how long before multiple-faces fires |
| `rules.absence_duration_sec` | 8.0 | how long before absence fires |
| `rules.object_min_confidence` | 0.6 | minimum confidence for a prohibited object |
| `rules.object_cooldown_sec` | 10.0 | seconds the object must be gone before it can fire again |

Detection is deliberately generous (low thresholds) and the rule engine is strict (0.6 and a sustained condition).

## Project structure

```
.
├── config.yaml          all tunable thresholds
├── requirements.txt     pipeline dependencies
├── run_pipeline.py      main pipeline entry point
├── dashboard.py         Streamlit review UI (own environment)
├── evaluate.py          false-positive and review report
├── start_demo.bat       Windows launcher
├── start_demo.ps1       starts pipeline and dashboard together
├── src/
│   ├── capture.py       video capture
│   ├── perception.py    face landmarks, head pose, gaze zone
│   ├── privacy.py       fail-closed anonymisation
│   ├── detection.py     YOLOv8-nano object detection
│   ├── rules.py         temporal rule engine
│   ├── explainability.py  explanation text
│   ├── logger.py        event and snapshot logging
│   └── config.py        loads config.yaml
└── data/
    ├── events/          JSON and blurred snapshots (not in git)
    └── metrics/         performance CSVs (not in git)
```

## Known limitations

These were found during testing and are left as they are.

- Head pitch ambiguity: six nearly flat landmarks give a pose that can flip by about 180 degrees. `perception.py` corrects this every frame (pitch above 90 has 180 subtracted, pitch below -90 has 180 added). It keeps no state, so it cannot drift.
- Iris height is unreliable: the eyelid hides the iris when looking down, so up and down gaze uses head pitch against the calibrated baseline. Left and right gaze uses the iris position, which worked well. Small eye-only glances are less reliable than head turns.
- Phone and laptop confusion: YOLOv8-nano sometimes mixes the two, which the label grouping handles.
- Hand false positive: an empty hand has been classified as a cell phone at over 60% confidence. A phone with its screen off, or shown from the back, is also detected less reliably. The proper fix is fine-tuning on staged images that include empty-hand examples, which has not been done.
- Duplicate events (fixed): one continuous phone use used to produce 12 events. A cooldown that needs the object to be gone for 10 seconds, plus label grouping, fixed this.
- One camera only: a second person outside the camera view is not seen.
- Testing was done by the author on one laptop, with no real exam dataset.
- No audio detection. It is not required by the BAI-06 brief and brings its own privacy problems.

## Built with

[MediaPipe](https://github.com/google-ai-edge/mediapipe) (Face Mesh and iris landmarks), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (pretrained COCO weights, not retrained), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [PyYAML](https://pyyaml.org/) and [Streamlit](https://streamlit.io/). The detectors are used as pretrained models. The privacy layer, head-pose and gaze logic, rule engine, explanations, logging, dashboard and evaluation are written for this project.