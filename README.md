# Exam Integrity Monitor — Setup

## One-time setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step 2a test
```bash
python src/capture.py
```
A window should open showing your live webcam feed. Press `q` to close it.
