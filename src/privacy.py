"""
privacy.py
Anonymization module with fail-safe (fail-closed) tracking: a brief
loss of face tracking must NOT reveal raw video.
"""
import cv2
from typing import List, Tuple, Optional


def _bounding_box(landmarks_px, w, h, padding: float = 0.15):
    xs = [p[0] for p in landmarks_px]
    ys = [p[1] for p in landmarks_px]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)
    return x1, y1, x2, y2


class PrivacyGuard:
    def __init__(self, grace_frames: int = 10):
        self.grace_frames = grace_frames
        self._last_boxes: List[Tuple[int, int, int, int]] = []
        self._frames_since_seen = 0

    def anonymize(self, frame, all_face_landmarks_px):
        h, w = frame.shape[:2]
        anonymized = frame.copy()

        if all_face_landmarks_px:
            boxes = [_bounding_box(lm, w, h) for lm in all_face_landmarks_px]
            self._last_boxes = boxes
            self._frames_since_seen = 0
        elif self._last_boxes and self._frames_since_seen < self.grace_frames:
            boxes = self._last_boxes
            self._frames_since_seen += 1
        else:
            boxes = None

        if boxes is None:
            k = (min(w, h) // 8) | 1
            return cv2.GaussianBlur(anonymized, (k, k), 0), []

        for (x1, y1, x2, y2) in boxes:
            face_region = anonymized[y1:y2, x1:x2]
            if face_region.size == 0:
                continue
            k = max(15, (min(x2 - x1, y2 - y1) // 3) | 1)
            anonymized[y1:y2, x1:x2] = cv2.GaussianBlur(face_region, (k, k), 0)

        return anonymized, boxes
