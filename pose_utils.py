"""
pose_utils.py
Utility functions for pose analysis: angle calculation, distance, feedback.
"""

import numpy as np


# ─── COCO Keypoint Index Map ───────────────────────────────────────────────
KP = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7,   "right_elbow": 8,
    "left_wrist": 9,   "right_wrist": 10,
    "left_hip": 11,    "right_hip": 12,
    "left_knee": 13,   "right_knee": 14,
    "left_ankle": 15,  "right_ankle": 16,
}


def get_keypoint(keypoints, name: str):
    """Return (x, y, conf) for a named keypoint; conf=0 if not found."""
    idx = KP.get(name)
    if idx is None or keypoints is None:
        return None
    kp = keypoints[idx]
    if len(kp) >= 2:
        x, y = float(kp[0]), float(kp[1])
        conf = float(kp[2]) if len(kp) > 2 else 1.0
        return (x, y, conf)
    return None


def calculate_angle(a, b, c) -> float:
    """
    Calculate the angle at joint b formed by points a-b-c.
    Each point is (x, y) or (x, y, conf).
    Returns angle in degrees [0, 180].
    """
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


def midpoint(a, b):
    """Return midpoint between two (x, y[, conf]) points."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def euclidean_distance(a, b) -> float:
    """Euclidean distance between two (x, y) points."""
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def is_visible(kp, threshold: float = 0.3) -> bool:
    """Check if a keypoint is visible above confidence threshold."""
    if kp is None:
        return False
    if len(kp) > 2:
        return kp[2] >= threshold
    return True


def all_visible(*kps, threshold: float = 0.3) -> bool:
    """Return True only if ALL keypoints are visible."""
    return all(is_visible(k, threshold) for k in kps)
