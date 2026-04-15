import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)


def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if result.pose_landmarks:
        return result.pose_landmarks
    return None


def draw_skeleton(frame, landmarks):
    mp_draw.draw_landmarks(
        frame,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
    )


def get_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def extract_angles(landmarks):
    lm = landmarks.landmark
    L = mp_pose.PoseLandmark

    return {
        "left_knee":      get_angle(lm[L.LEFT_HIP],       lm[L.LEFT_KNEE],      lm[L.LEFT_ANKLE]),
        "right_knee":     get_angle(lm[L.RIGHT_HIP],      lm[L.RIGHT_KNEE],     lm[L.RIGHT_ANKLE]),
        "left_hip":       get_angle(lm[L.LEFT_SHOULDER],  lm[L.LEFT_HIP],       lm[L.LEFT_KNEE]),
        "right_hip":      get_angle(lm[L.RIGHT_SHOULDER], lm[L.RIGHT_HIP],      lm[L.RIGHT_KNEE]),
        "left_elbow":     get_angle(lm[L.LEFT_SHOULDER],  lm[L.LEFT_ELBOW],     lm[L.LEFT_WRIST]),
        "right_elbow":    get_angle(lm[L.RIGHT_SHOULDER], lm[L.RIGHT_ELBOW],    lm[L.RIGHT_WRIST]),
        "left_shoulder":  get_angle(lm[L.LEFT_ELBOW],     lm[L.LEFT_SHOULDER],  lm[L.LEFT_HIP]),
        "right_shoulder": get_angle(lm[L.RIGHT_ELBOW],    lm[L.RIGHT_SHOULDER], lm[L.RIGHT_HIP]),
    }


def is_visible(landmarks, indices, threshold=0.6):
    lm = landmarks.landmark
    return all(lm[i].visibility > threshold for i in indices)