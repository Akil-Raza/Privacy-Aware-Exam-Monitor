"""
perception.py
Face landmarks + head-pose + hybrid gaze zone: iris dx for left/right
(confirmed reliable), head pitch relative to a per-session calibrated
baseline for up/down (iris dy alone could not separate "looking down"
from resting eyelid occlusion in testing).
"""
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float


@dataclass
class GazeZone:
    zone: str
    dx: float
    dy: float


@dataclass
class PerceptionResult:
    num_faces: int
    head_pose: Optional[HeadPose]
    gaze_zone: Optional[GazeZone]
    primary_landmarks_px: Optional[List[Tuple[int, int]]]
    all_face_landmarks_px: List[List[Tuple[int, int]]]


_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)
_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

_LEFT_IRIS_CENTER = 468
_LEFT_EYE_CORNERS = (33, 133)
_RIGHT_IRIS_CENTER = 473
_RIGHT_EYE_CORNERS = (362, 263)


def _eye_offset(landmarks_px, iris_idx, corner_a, corner_b):
    iris = landmarks_px[iris_idx]
    a, b = landmarks_px[corner_a], landmarks_px[corner_b]
    center_x, center_y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    eye_width = abs(a[0] - b[0])
    if eye_width == 0:
        return 0.0, 0.0
    return (iris[0] - center_x) / eye_width, (iris[1] - center_y) / eye_width


class FaceAnalyzer:
    def __init__(self, max_num_faces: int = 3, pitch_calibration_frames: int = 30):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self._pitch_calibration_frames = pitch_calibration_frames
        self._pitch_samples = []
        self._pitch_baseline = None

    def analyze(self, frame) -> PerceptionResult:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return PerceptionResult(0, None, None, None, [])

        all_face_landmarks_px = [
            [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]
            for face in results.multi_face_landmarks
        ]
        primary = all_face_landmarks_px[0]
        head_pose = self._estimate_head_pose(primary, w, h)
        if head_pose is not None:
            self._update_calibration(head_pose.pitch)
        gaze_zone = self._estimate_gaze_zone(primary, head_pose)
        return PerceptionResult(len(all_face_landmarks_px), head_pose, gaze_zone, primary, all_face_landmarks_px)

    def _update_calibration(self, pitch: float):
        if self._pitch_baseline is not None:
            return
        self._pitch_samples.append(pitch)
        if len(self._pitch_samples) >= self._pitch_calibration_frames:
            self._pitch_baseline = float(np.mean(self._pitch_samples))

    def _estimate_gaze_zone(self, landmarks_px, head_pose: Optional[HeadPose]) -> Optional[GazeZone]:
        try:
            dx1, dy1 = _eye_offset(landmarks_px, _LEFT_IRIS_CENTER, *_LEFT_EYE_CORNERS)
            dx2, dy2 = _eye_offset(landmarks_px, _RIGHT_IRIS_CENTER, *_RIGHT_EYE_CORNERS)
        except IndexError:
            return None
        dx, dy = (dx1 + dx2) / 2, (dy1 + dy2) / 2

        relative_pitch = None
        if head_pose and self._pitch_baseline is not None:
            relative_pitch = head_pose.pitch - self._pitch_baseline

        if dx < -0.15:
            zone = "left"
        elif dx > 0.15:
            zone = "right"
        elif relative_pitch is not None and relative_pitch > 8:
            zone = "down"
        elif relative_pitch is not None and relative_pitch < -8:
            zone = "up"
        else:
            zone = "center"
        return GazeZone(zone=zone, dx=dx, dy=dy)

    def _estimate_head_pose(self, landmarks_px, w, h) -> Optional[HeadPose]:
        try:
            image_points = np.array([landmarks_px[i] for i in _LANDMARK_IDS], dtype=np.float64)
        except IndexError:
            return None
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vec, _ = cv2.solvePnP(_MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_mat)
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)

    @staticmethod
    def _rotation_matrix_to_euler(R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees(x), np.degrees(y), np.degrees(z)

    def close(self):
        self._mesh.close()
