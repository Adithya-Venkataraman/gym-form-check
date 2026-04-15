import cv2
import sys
import mediapipe as mp
from pose_utils import get_landmarks, draw_skeleton, extract_angles, is_visible
from rules import check_form, detect_exercise, make_tracker, EXERCISES

L = mp.solutions.pose.PoseLandmark

KEY_LANDMARKS = {
    "squat":               [L.LEFT_HIP,      L.LEFT_KNEE,  L.LEFT_ANKLE],
    "pushup":              [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST],
    "curl":                [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST],
    "incline_chest_press": [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST],
}


def draw_feedback(frame, feedback, exercise, reps, angles=None):
    h, w = frame.shape[:2]

    # Exercise label
    label = exercise.replace("_", " ").upper() if exercise else "DETECTING..."
    color = (0, 200, 255) if exercise else (100, 100, 255)
    cv2.putText(frame, f"Exercise: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Rep count
    if exercise:
        cv2.putText(frame, f"Auto-detected  |  Reps: {reps}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # Quit hint
    cv2.putText(frame, "Q = quit", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Feedback at TOP right below the header
    if feedback:
        box_bottom = 80 + len(feedback[:4]) * 38
        cv2.rectangle(frame, (0, 70), (w, box_bottom), (0, 0, 0), -1)
        for i, line in enumerate(feedback[:4]):
            y = 100 + i * 38
            c = (0, 255, 100) if "Good" in line else (0, 100, 255)
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print("Cannot open source:", src)
        return

    trackers       = {ex: make_tracker(ex) for ex in EXERCISES}
    last_feedback  = []
    last_exercise  = None
    feedback_timer = 0
    angles         = None

    print("Running — Q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame     = cv2.resize(frame, (540, 960))
        landmarks = get_landmarks(frame)
        exercise  = None
        reps      = 0

        if landmarks:
            draw_skeleton(frame, landmarks)
            angles   = extract_angles(landmarks)
            exercise = detect_exercise(angles, landmarks)

            if exercise and is_visible(landmarks, [i.value for i in KEY_LANDMARKS[exercise]]):
                if exercise != last_exercise:
                    trackers[exercise] = make_tracker(exercise)
                    last_feedback  = []
                    feedback_timer = 0

                new_feedback = check_form(exercise, angles, trackers[exercise])

                # only update when feedback actually arrives
                if new_feedback:
                    last_feedback  = new_feedback
                    feedback_timer = 90  # hold for 90 frames (~3 seconds)

                last_exercise = exercise
                reps          = trackers[exercise].reps

        # tick down timer — only clear feedback after it expires
        if feedback_timer > 0:
            feedback_timer -= 1
        elif feedback_timer == 0:
            last_feedback = []

        draw_feedback(frame, last_feedback, exercise, reps, angles)
        cv2.imshow("Gym Form Checker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()