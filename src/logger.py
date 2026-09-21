"""
logger.py
Event logger: writes each event as JSON + its anonymized snapshot.
"""
import cv2
import json
import os
import time
from dataclasses import asdict


class EventLogger:
    def __init__(self, output_dir: str = "data/events"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def log(self, event, anonymized_frame):
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.triggered_at))
        base_name = f"{timestamp_str}_{event.rule_id}"
        image_filename = f"{base_name}.jpg"
        cv2.imwrite(os.path.join(self.output_dir, image_filename), anonymized_frame)
        record = asdict(event)
        record["snapshot"] = image_filename
        with open(os.path.join(self.output_dir, f"{base_name}.json"), "w") as f:
            json.dump(record, f, indent=2)
        print(f"[EVENT LOGGED] {event.rule_id}: {event.description}")
