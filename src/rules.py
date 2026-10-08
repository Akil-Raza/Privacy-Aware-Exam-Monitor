"""
rules.py
Temporal rule engine. Every rule needs a SUSTAINED condition before firing.

GazeAwayRule considers EITHER head yaw OR gaze zone off-center, catching
the case where the head stays forward but the eyes move.

The gaze, multiple-faces and absence rules tolerate very short dropouts
(dropout_tolerance_sec). Real camera signals flicker: one frame where the
second face is missed, or yaw dipping just under the threshold. Without a
tolerance, a single bad frame reset the timer and the rule never fired.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from explainability import ExplainabilityEngine


@dataclass
class RuleEvent:
    rule_id: str
    description: str
    confidence: float
    triggered_at: float
    details: Dict = field(default_factory=dict)


class SustainedCondition:
    """Measures how long a condition has been true, forgiving short dropouts."""

    def __init__(self, tolerance_s: float = 1.0):
        self.tolerance_s = tolerance_s
        self.since = None
        self.last_true = None

    def update(self, active: bool, now: float) -> Optional[float]:
        """Returns seconds the condition has held, or None if it is over."""
        if active:
            if self.since is None:
                self.since = now
            self.last_true = now
            return now - self.since
        if self.since is not None and now - self.last_true <= self.tolerance_s:
            return now - self.since  # brief dropout, keep timing
        self.since = None
        self.last_true = None
        return None


class GazeAwayRule:
    def __init__(self, yaw_threshold: float = 25.0, duration_s: float = 5.0, tolerance_s: float = 1.0):
        self.yaw_threshold = yaw_threshold
        self.duration_s = duration_s
        self._timer = SustainedCondition(tolerance_s)
        self._fired = False
        self._away_reason = None

    def update(self, yaw: Optional[float], gaze_zone: Optional[str] = None) -> Optional[RuleEvent]:
        now = time.time()
        head_turned = yaw is not None and abs(yaw) > self.yaw_threshold
        eyes_off = gaze_zone is not None and gaze_zone != "center"
        looking_away = head_turned or eyes_off

        was_idle = self._timer.since is None
        elapsed = self._timer.update(looking_away, now)
        if elapsed is None:
            self._fired = False
            self._away_reason = None
            return None
        if was_idle:
            self._away_reason = "head_turn" if head_turned else "eye_gaze"

        if looking_away and elapsed >= self.duration_s and not self._fired:
            self._fired = True
            details = {"yaw": yaw, "gaze_zone": gaze_zone, "reason": self._away_reason, "duration_sec": self.duration_s}
            confidence = min(1.0, elapsed / (self.duration_s * 2))
            return RuleEvent("GAZE_AWAY", ExplainabilityEngine.generate("GAZE_AWAY", confidence, details), confidence, now, details)
        return None


class MultipleFacesRule:
    def __init__(self, duration_s: float = 3.0, tolerance_s: float = 1.0):
        self.duration_s = duration_s
        self._timer = SustainedCondition(tolerance_s)
        self._fired = False

    def update(self, num_faces: int) -> Optional[RuleEvent]:
        now = time.time()
        elapsed = self._timer.update(num_faces >= 2, now)
        if elapsed is None:
            self._fired = False
            return None
        if num_faces >= 2 and elapsed >= self.duration_s and not self._fired:
            self._fired = True
            details = {"num_faces": num_faces, "duration_sec": self.duration_s}
            return RuleEvent("MULTIPLE_FACES", ExplainabilityEngine.generate("MULTIPLE_FACES", 0.9, details), 0.9, now, details)
        return None


class ProlongedAbsenceRule:
    def __init__(self, duration_s: float = 8.0, tolerance_s: float = 1.0):
        self.duration_s = duration_s
        self._timer = SustainedCondition(tolerance_s)
        self._fired = False

    def update(self, num_faces: int) -> Optional[RuleEvent]:
        now = time.time()
        elapsed = self._timer.update(num_faces == 0, now)
        if elapsed is None:
            self._fired = False
            return None
        if num_faces == 0 and elapsed >= self.duration_s and not self._fired:
            self._fired = True
            details = {"duration_sec": self.duration_s}
            return RuleEvent("PROLONGED_ABSENCE", ExplainabilityEngine.generate("PROLONGED_ABSENCE", 0.85, details), 0.85, now, details)
        return None


class ProhibitedObjectRule:
    def __init__(self, label_groups=None, min_confidence: float = 0.6, cooldown_s: float = 10.0):
        self.label_groups = label_groups or {"cell phone": "device", "laptop": "device", "book": "book"}
        self.min_confidence = min_confidence
        self.cooldown_s = cooldown_s
        self._last_seen = {}
        self._already_fired = set()

    def update(self, detections) -> List[RuleEvent]:
        now = time.time()
        events = []
        for det in detections:
            group = self.label_groups.get(det.label)
            if group is None or det.confidence < self.min_confidence:
                continue
            self._last_seen[group] = now
            if group not in self._already_fired:
                self._already_fired.add(group)
                details = {"label": det.label, "group": group}
                events.append(RuleEvent("PROHIBITED_OBJECT", ExplainabilityEngine.generate("PROHIBITED_OBJECT", det.confidence, details), det.confidence, now, details))
        for group in list(self._already_fired):
            if now - self._last_seen.get(group, 0) > self.cooldown_s:
                self._already_fired.discard(group)
        return events


class RuleEngine:
    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        tol = cfg.get("dropout_tolerance_sec", 1.0)
        self.gaze_rule = GazeAwayRule(cfg.get("gaze_yaw_threshold", 25.0), cfg.get("gaze_duration_sec", 5.0), tol)
        self.faces_rule = MultipleFacesRule(cfg.get("multi_face_duration_sec", 3.0), tol)
        self.absence_rule = ProlongedAbsenceRule(cfg.get("absence_duration_sec", 8.0), tol)
        self.object_rule = ProhibitedObjectRule(
            label_groups=cfg.get("label_groups", {"cell phone": "device", "laptop": "device", "book": "book"}),
            min_confidence=cfg.get("object_min_confidence", 0.6),
            cooldown_s=cfg.get("object_cooldown_sec", 10.0),
        )

    def update(self, *, yaw, gaze_zone, num_faces, detections) -> List[RuleEvent]:
        events = []
        for event in (self.gaze_rule.update(yaw, gaze_zone), self.faces_rule.update(num_faces), self.absence_rule.update(num_faces)):
            if event:
                events.append(event)
        events.extend(self.object_rule.update(detections))
        return events