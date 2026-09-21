"""
capture.py
Step 2a — Local video capture module.
"""
import cv2
import time
from typing import Generator, Optional, Union


class FrameSource:
    def __init__(self, source: Union[int, str] = 0, target_fps: int = 15):
        self.source = source
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

    def frames(self) -> Generator["cv2.Mat", None, None]:
        if self.cap is None:
            self.open()
        frame_interval = 1.0 / self.target_fps
        last_time = 0.0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            now = time.time()
            if now - last_time < frame_interval:
                continue
            last_time = now
            yield frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


if __name__ == "__main__":
    source = FrameSource(source=0, target_fps=15)
    try:
        for frame in source.frames():
            cv2.putText(frame, "Step 2a: raw capture test - press q to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Exam Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
