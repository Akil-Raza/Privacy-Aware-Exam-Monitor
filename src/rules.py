"""
rules.py
ProhibitedObjectRule now groups visually-similar labels (a small
detector can flicker between labels for the SAME physical object,
e.g. phone read as "laptop" one frame, "cell phone" the next) so one
real appearance only ever fires one event.
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


class GazeAwayRule:
    def __init__(self, yaw_threshold: float = 25.0, duration_s: float = 5.0):
        self.yaw_threshold = yaw_threshold
        self.duration_s = duration_s
        self._away_since = None
        self._fired = False

    def update(self, yaw: Optional[float]) -> Optional[RuleEvent]:
        now = time.time()
        looking_away = yaw is not None and abs(yaw) > self.yaw_threshold
        if not looking_away:
            self._away_since = None
            self._fired = False
            return None
        if self._away_since is None:
            self._away_since = now
        elapsed = now - self._away_since
        if elapsed >= self.duration_s and not self._fired:
            self._fired = True
            details = {"yaw": yaw, "duration_sec": self.duration_s}
            confidence = min(1.0, elapsed / (self.duration_s * 2))
            return RuleEvent("GAZE_AWAY", ExplainabilityEngine.generate("GAZE_AWAY", confidence, details), confidence, now, details)
        return None


class MultipleFacesRule:
    def __init__(self, duration_s: float = 3.0):
        self.duration_s = duration_s
        self._since = None
        self._fired = False

    def update(self, num_faces: int) -> Optional[RuleEvent]:
        now = time.time()
        if num_faces <= 1:
            self._since = None
            self._fired = False
            return None
        if self._since is None:
            self._since = now
        elapsed = now - self._since
        if elapsed >= self.duration_s and not self._fired:
            self._fired = True
            details = {"num_faces": num_faces, "duration_sec": self.duration_s}
            return RuleEvent("MULTIPLE_FACES", ExplainabilityEngine.generate("MULTIPLE_FACES", 0.9, details), 0.9, now, details)
        return None


class ProlongedAbsenceRule:
    def __init__(self, duration_s: float = 8.0):
        self.duration_s = duration_s
        self._since = None
        self._fired = False

    def update(self, num_faces: int) -> Optional[RuleEvent]:
        now = time.time()
        if num_faces > 0:
            self._since = None
            self._fired = False
            return None
        if self._since is None:
            self._since = now
        elapsed = now - self._since
        if elapsed >= self.duration_s and not self._fired:
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
        self.gaze_rule = GazeAwayRule(cfg.get("gaze_yaw_threshold", 25.0), cfg.get("gaze_duration_sec", 5.0))
        self.faces_rule = MultipleFacesRule(cfg.get("multi_face_duration_sec", 3.0))
        self.absence_rule = ProlongedAbsenceRule(cfg.get("absence_duration_sec", 8.0))
        self.object_rule = ProhibitedObjectRule(
            label_groups=cfg.get("label_groups", {"cell phone": "device", "laptop": "device", "book": "book"}),
            min_confidence=cfg.get("object_min_confidence", 0.6),
            cooldown_s=cfg.get("object_cooldown_sec", 10.0),
        )

    def update(self, *, yaw, num_faces, detections) -> List[RuleEvent]:
        events = []
        for event in (self.gaze_rule.update(yaw), self.faces_rule.update(num_faces), self.absence_rule.update(num_faces)):
            if event:
                events.append(event)
        events.extend(self.object_rule.update(detections))
        return events
