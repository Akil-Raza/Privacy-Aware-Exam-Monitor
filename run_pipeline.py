"""
run_pipeline.py
Full pipeline. Evidence overlays are drawn BEFORE logging, so saved
snapshots include the boxes that justify the alert, not a plain blur.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
from capture import FrameSource
from perception import FaceAnalyzer
from privacy import PrivacyGuard
from detection import ObjectDetector
from rules import RuleEngine
from logger import EventLogger


def main():
    source = FrameSource(source=0, target_fps=10)
    analyzer = FaceAnalyzer(max_num_faces=3)
    guard = PrivacyGuard(grace_frames=10)
    detector = ObjectDetector(confidence_threshold=0.5)
    rule_engine = RuleEngine()
    event_logger = EventLogger(output_dir="data/events")

    print("Pipeline running. Press 'q' in the window to quit.\n")
    try:
        for frame in source.frames():
            perception = analyzer.analyze(frame)
            detections = detector.detect(frame)
            anonymized, boxes = guard.anonymize(frame, perception.all_face_landmarks_px)

            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(anonymized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for det in detections:
                x1, y1, x2, y2 = det.box
                cv2.rectangle(anonymized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(anonymized, det.label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            yaw = perception.head_pose.yaw if perception.head_pose else None
            events = rule_engine.update(yaw=yaw, num_faces=perception.num_faces, detections=detections)
            for event in events:
                event_logger.log(event, anonymized)  # now includes the boxes

            status = f"Faces: {perception.num_faces}"
            if yaw is not None:
                status += f"  Yaw: {yaw:.1f}"
            cv2.putText(anonymized, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Exam Integrity Monitor - full pipeline", anonymized)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        analyzer.close()
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
