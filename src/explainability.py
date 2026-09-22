"""
explainability.py
Deterministic, proctor-readable explanation text per rule type.
GAZE_AWAY now distinguishes head-turn vs eyes-only causes.
"""
from typing import Dict


class ExplainabilityEngine:
    @staticmethod
    def generate(rule_id: str, confidence: float, details: Dict) -> str:
        generators = {
            "GAZE_AWAY": ExplainabilityEngine._gaze_away,
            "MULTIPLE_FACES": ExplainabilityEngine._multiple_faces,
            "PROLONGED_ABSENCE": ExplainabilityEngine._prolonged_absence,
            "PROHIBITED_OBJECT": ExplainabilityEngine._prohibited_object,
        }
        fn = generators.get(rule_id, ExplainabilityEngine._generic)
        return fn(confidence, details)

    @staticmethod
    def _gaze_away(confidence, details):
        duration = details.get("duration_sec", 0.0)
        reason = details.get("reason", "head_turn")
        if reason == "eye_gaze":
            zone = details.get("gaze_zone", "away from center")
            return f"Student's eyes were directed {zone} (head remained forward) continuously for over {duration:.0f} seconds. Confidence: {confidence:.0%}."
        yaw = details.get("yaw", 0.0)
        direction = "left" if yaw < 0 else "right"
        return f"Student's head was turned to the {direction} (yaw {abs(yaw):.0f} deg) continuously for over {duration:.0f} seconds. Confidence: {confidence:.0%}."

    @staticmethod
    def _multiple_faces(confidence, details):
        count = details.get("num_faces", 0)
        duration = details.get("duration_sec", 0.0)
        return f"{count} faces were visible in frame (expected 1) for over {duration:.0f} seconds - possibly a second person present. Confidence: {confidence:.0%}."

    @staticmethod
    def _prolonged_absence(confidence, details):
        duration = details.get("duration_sec", 0.0)
        return f"No face was detected for over {duration:.0f} seconds - the student may have left the frame, or the camera view is obstructed. Confidence: {confidence:.0%}."

    @staticmethod
    def _prohibited_object(confidence, details):
        label = details.get("label", "an unrecognized object")
        return f"A {label} was detected in frame. Confidence: {confidence:.0%}."

    @staticmethod
    def _generic(confidence, details):
        return f"Rule triggered. Details: {details}. Confidence: {confidence:.0%}."
