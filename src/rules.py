"""
rules.py
Temporal rule engine. ProhibitedObjectRule now requires a TRUE sustained
absence (cooldown) before re-firing, fixing duplicate-event inflation
from single-frame detector flicker on a continuously-present object.
"""
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RuleEvent:
    rule_id: str
    description: str
    confidence: float
    triggered_at: float


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
            return RuleEvent("GAZE_AWAY", f"Gaze away (yaw {yaw:.1f} deg) for over {self.duration_s:.0f}s",
                              min(1.0, elapsed / (self.duration_s * 2)), now)
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
            return RuleEvent("MULTIPLE_FACES", f"{num_faces} faces detected for over {self.duration_s:.0f}s", 0.9, now)
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
            return RuleEvent("PROLONGED_ABSENCE", f"No face detected for over {self.duration_s:.0f}s", 0.85, now)
        return None


class ProhibitedObjectRule:
    def __init__(self, watched_labels=("cell phone", "book"), min_confidence: float = 0.6, cooldown_s: float = 10.0):
        self.watched_labels = set(watched_labels)
        self.min_confidence = min_confidence
        self.cooldown_s = cooldown_s
        self._last_seen = {}
        self._already_fired = set()

    def update(self, detections) -> List[RuleEvent]:
        now = time.time()
        events = []
        for det in detections:
            if det.label not in self.watched_labels or det.confidence < self.min_confidence:
                continue
            self._last_seen[det.label] = now
            if det.label not in self._already_fired:
                self._already_fired.add(det.label)
                events.append(RuleEvent("PROHIBITED_OBJECT", f"{det.label} detected (confidence {det.confidence:.2f})", det.confidence, now))
        for label in list(self._already_fired):
            if now - self._last_seen.get(label, 0) > self.cooldown_s:
                self._already_fired.discard(label)
        return events


class RuleEngine:
    def __init__(self):
        self.gaze_rule = GazeAwayRule()
        self.faces_rule = MultipleFacesRule()
        self.absence_rule = ProlongedAbsenceRule()
        self.object_rule = ProhibitedObjectRule()

    def update(self, *, yaw, num_faces, detections) -> List[RuleEvent]:
        events = []
        for event in (self.gaze_rule.update(yaw), self.faces_rule.update(num_faces), self.absence_rule.update(num_faces)):
            if event:
                events.append(event)
        events.extend(self.object_rule.update(detections))
        return events
