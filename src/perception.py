"""
perception.py
Face landmark detection + head-pose estimation (multi-face aware).
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
class PerceptionResult:
    num_faces: int
    head_pose: Optional[HeadPose]
    primary_landmarks_px: Optional[List[Tuple[int, int]]]
    all_face_landmarks_px: List[List[Tuple[int, int]]]


_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)

_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


class FaceAnalyzer:
    def __init__(self, max_num_faces: int = 3):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    def analyze(self, frame) -> PerceptionResult:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return PerceptionResult(0, None, None, [])

        all_face_landmarks_px = [
            [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]
            for face in results.multi_face_landmarks
        ]
        primary = all_face_landmarks_px[0]
        head_pose = self._estimate_head_pose(primary, w, h)
        return PerceptionResult(len(all_face_landmarks_px), head_pose, primary, all_face_landmarks_px)

    def _estimate_head_pose(self, landmarks_px, w, h) -> Optional[HeadPose]:
        try:
            image_points = np.array([landmarks_px[i] for i in _LANDMARK_IDS], dtype=np.float64)
        except IndexError:
            return None
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vec, _ = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_mat)
        # 6-point model is near-planar: correct the known +/-180 pitch ambiguity.
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


if __name__ == "__main__":
    from capture import FrameSource
    source = FrameSource(source=0, target_fps=15)
    analyzer = FaceAnalyzer(max_num_faces=3)
    try:
        for frame in source.frames():
            result = analyzer.analyze(frame)
            cv2.putText(frame, f"Faces: {result.num_faces}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if result.head_pose:
                hp = result.head_pose
                cv2.putText(frame, f"Yaw:{hp.yaw:.1f} Pitch:{hp.pitch:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("perception test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        analyzer.close()
        source.release()
        cv2.destroyAllWindows()
