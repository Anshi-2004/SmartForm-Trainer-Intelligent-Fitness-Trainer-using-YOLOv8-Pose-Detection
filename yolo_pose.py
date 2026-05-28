"""
yolo_pose.py
YOLO model wrapper – loads the model once and provides
keypoint extraction + skeleton drawing utilities.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st

# ─── Skeleton pairs (COCO 17-keypoint format) ─────────────────────────────
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # face
    (5, 6),                                  # shoulders
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 11), (6, 12),                        # torso sides
    (11, 12),                                # hips
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

JOINT_COLORS = {
    "face":   (255, 200, 100),
    "arm":    (100, 200, 255),
    "torso":  (100, 255, 180),
    "leg":    (255, 130, 100),
}

PAIR_COLORS = [
    JOINT_COLORS["face"],   # (0,1)
    JOINT_COLORS["face"],   # (0,2)
    JOINT_COLORS["face"],   # (1,3)
    JOINT_COLORS["face"],   # (2,4)
    JOINT_COLORS["torso"],  # (5,6)
    JOINT_COLORS["arm"],    # (5,7)
    JOINT_COLORS["arm"],    # (7,9)
    JOINT_COLORS["arm"],    # (6,8)
    JOINT_COLORS["arm"],    # (8,10)
    JOINT_COLORS["torso"],  # (5,11)
    JOINT_COLORS["torso"],  # (6,12)
    JOINT_COLORS["torso"],  # (11,12)
    JOINT_COLORS["leg"],    # (11,13)
    JOINT_COLORS["leg"],    # (13,15)
    JOINT_COLORS["leg"],    # (12,14)
    JOINT_COLORS["leg"],    # (14,16)
]

KEYPOINT_COLOR = (80, 230, 255)
KEYPOINT_RADIUS = 5
BONE_THICKNESS = 2


@st.cache_resource(show_spinner="Loading YOLO Pose model…")
def load_model(weights: str = "yolov8n-pose.pt") -> YOLO:
    """Load and cache the YOLO pose model."""
    model = YOLO(weights)
    return model


def extract_keypoints(frame: np.ndarray, model: YOLO, conf: float = 0.5):
    """
    Run inference on a single BGR frame.
    Returns:
        keypoints: np.ndarray of shape (17, 3) [x, y, conf] for the first person,
                   or None if no person detected.
        annotated: BGR frame with skeleton overlay drawn.
    """
    results = model(frame, verbose=False, conf=conf)
    annotated = frame.copy()

    if results and len(results) > 0:
        result = results[0]
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            # Take the first detected person
            kps_tensor = result.keypoints.data[0]  # shape (17, 3)
            kps = kps_tensor.cpu().numpy()
            annotated = _draw_skeleton(annotated, kps)
            return kps, annotated

    return None, annotated


def _draw_skeleton(frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Draw skeleton lines and keypoint circles on the frame."""
    h, w = frame.shape[:2]

    # Draw bones
    for idx, (i, j) in enumerate(SKELETON_PAIRS):
        xi, yi, ci = kps[i]
        xj, yj, cj = kps[j]
        if ci > 0.3 and cj > 0.3:
            pt1 = (int(xi), int(yi))
            pt2 = (int(xj), int(yj))
            color = PAIR_COLORS[idx] if idx < len(PAIR_COLORS) else (200, 200, 200)
            cv2.line(frame, pt1, pt2, color, BONE_THICKNESS, cv2.LINE_AA)

    # Draw joints
    for x, y, c in kps:
        if c > 0.3:
            cv2.circle(frame, (int(x), int(y)), KEYPOINT_RADIUS,
                       KEYPOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), KEYPOINT_RADIUS + 1,
                       (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_angle_arc(frame: np.ndarray, pt: tuple, angle: float,
                   color=(255, 255, 255)):
    """Draw the angle value near a joint."""
    x, y = int(pt[0]), int(pt[1])
    cv2.putText(frame, f"{int(angle)}°",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"{int(angle)}°",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
