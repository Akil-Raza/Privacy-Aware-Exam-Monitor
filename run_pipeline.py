"""
run_pipeline.py
Config-driven full pipeline. Rule engine now receives BOTH head yaw and
iris/pitch-based gaze zone, catching eyes-only looking-away too.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import time
import csv
import statistics as stats
from collections import defaultdict

from config import load_config
from capture import FrameSource
from perception import FaceAnalyzer
from privacy import PrivacyGuard
from detection import ObjectDetector
from rules import RuleEngine
from logger import EventLogger


def summarize(times):
    if not times:
        return {"mean_ms": 0, "min_ms": 0, "max_ms": 0, "p95_ms": 0}
    times_ms = sorted(t * 1000 for t in times)
    p95_index = min(int(len(times_ms) * 0.95), len(times_ms) - 1)
    return {"mean_ms": round(stats.mean(times_ms), 2), "min_ms": round(min(times_ms), 2),
            "max_ms": round(max(times_ms), 2), "p95_ms": round(times_ms[p95_index], 2)}


def main():
    cfg = load_config()

    source = FrameSource(source=cfg["capture"]["source"], target_fps=cfg["capture"]["target_fps"])
    analyzer = FaceAnalyzer(max_num_faces=3)
    guard = PrivacyGuard(grace_frames=cfg["privacy"]["grace_frames"])
    detector = ObjectDetector(
        confidence_threshold=cfg["detection"]["confidence_threshold"],
        class_overrides=cfg["detection"]["class_overrides"],
    )
    rule_engine = RuleEngine(cfg["rules"])
    event_logger = EventLogger(output_dir=cfg["logging"]["output_dir"])

    stage_times = defaultdict(list)
    frame_times = []
    last_report = time.perf_counter()
    print("Pipeline running. Press 'q' in the window to quit.\n")

    try:
        for frame in source.frames():
            frame_start = time.perf_counter()

            t0 = time.perf_counter()
            perception = analyzer.analyze(frame)
            stage_times["perception"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            detections = detector.detect(frame)
            stage_times["detection"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            anonymized, boxes = guard.anonymize(frame, perception.all_face_landmarks_px)
            stage_times["anonymization"].append(time.perf_counter() - t0)

            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(anonymized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for det in detections:
                x1, y1, x2, y2 = det.box
                cv2.rectangle(anonymized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(anonymized, det.label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            t0 = time.perf_counter()
            yaw = perception.head_pose.yaw if perception.head_pose else None
            gaze_zone = perception.gaze_zone.zone if perception.gaze_zone else None
            events = rule_engine.update(yaw=yaw, gaze_zone=gaze_zone, num_faces=perception.num_faces, detections=detections)
            stage_times["rules"].append(time.perf_counter() - t0)

            for event in events:
                event_logger.log(event, anonymized)

            frame_times.append(time.perf_counter() - frame_start)
            status = f"Faces: {perception.num_faces}"
            if yaw is not None:
                status += f"  Yaw: {yaw:.1f}"
            if gaze_zone:
                status += f"  Gaze: {gaze_zone}"
            if frame_times[-1] > 0:
                status += f"  FPS: {1.0 / frame_times[-1]:.1f}"
            cv2.putText(anonymized, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Exam Integrity Monitor - full pipeline", anonymized)

            if time.perf_counter() - last_report > 5.0:
                recent = frame_times[-50:]
                avg_fps = len(recent) / sum(recent) if recent else 0
                print(f"[metrics] rolling FPS: {avg_fps:.1f}  |  perception: {stage_times['perception'][-1]*1000:.1f}ms  "
                      f"detection: {stage_times['detection'][-1]*1000:.1f}ms  anonymization: {stage_times['anonymization'][-1]*1000:.1f}ms")
                last_report = time.perf_counter()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        analyzer.close()
        source.release()
        cv2.destroyAllWindows()

        os.makedirs("data/metrics", exist_ok=True)
        summary_path = "data/metrics/performance_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "mean_ms", "min_ms", "max_ms", "p95_ms", "samples"])
            for stage_name, times in stage_times.items():
                s = summarize(times)
                writer.writerow([stage_name, s["mean_ms"], s["min_ms"], s["max_ms"], s["p95_ms"], len(times)])
            overall_fps = len(frame_times) / sum(frame_times) if frame_times else 0
            writer.writerow(["overall_fps", round(overall_fps, 2), "", "", "", len(frame_times)])
        print(f"\nSession ended. Performance summary written to {summary_path}")


if __name__ == "__main__":
    main()
