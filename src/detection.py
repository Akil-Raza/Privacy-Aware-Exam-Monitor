"""
detection.py
Object detection for unauthorized items using YOLOv8-nano.
"""
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple

_CLASSES_OF_INTEREST = {0: "person", 63: "laptop", 67: "cell phone", 73: "book"}


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]


class ObjectDetector:
    def __init__(self, confidence_threshold: float = 0.5, class_overrides: dict = None):
        self.model = YOLO("yolov8n.pt")
        self.confidence_threshold = confidence_threshold
        # Screen-off/back-facing phones score lower confidence on COCO-trained
        # weights — accept lower confidence specifically for that class.
        self.class_overrides = class_overrides or {"cell phone": 0.30}

    def detect(self, frame) -> List[Detection]:
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in _CLASSES_OF_INTEREST:
                continue
            label = _CLASSES_OF_INTEREST[cls_id]
            conf = float(box.conf[0])
            threshold = self.class_overrides.get(label, self.confidence_threshold)
            if conf < threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(Detection(label=label, confidence=conf, box=(x1, y1, x2, y2)))
        return detections


if __name__ == "__main__":
    import cv2
    from capture import FrameSource
    source = FrameSource(source=0, target_fps=10)
    detector = ObjectDetector(confidence_threshold=0.5)
    try:
        for frame in source.frames():
            for det in detector.detect(frame):
                x1, y1, x2, y2 = det.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{det.label} {det.confidence:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("detection test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
