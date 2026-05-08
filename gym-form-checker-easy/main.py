import cv2
import sys
import mediapipe as mp
from pose_utils import get_landmarks, draw_skeleton, extract_angles
from rules import check_form, detect_exercise, make_tracker, EXERCISES

L = mp.solutions.pose.PoseLandmark

KEY_LANDMARKS = {
    "squat": [
        [L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE],
        [L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE],
    ],
    "pushup": [
        [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST],
        [L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST],
    ],
    "curl": [
        [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST],
        [L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST],
    ],
    "incline_chest_press": [
        [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST],
        [L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST],
    ],
}


def has_required_visibility(landmarks, exercise, threshold=0.20):
    if exercise not in KEY_LANDMARKS:
        return False

    lm = landmarks.landmark
    if exercise == "incline_chest_press":
        threshold = 0.16

    for joint_set in KEY_LANDMARKS[exercise]:
        vis = [lm[j.value].visibility for j in joint_set]
        if (sum(vis) / len(vis)) > threshold and max(vis) > (threshold + 0.05):
            return True
    return False


def draw_feedback(frame, feedback, exercise, reps):
    h, w = frame.shape[:2]

    label = exercise.replace("_", " ").upper() if exercise else "DETECTING..."
    color = (0, 200, 255) if exercise else (100, 100, 255)
    cv2.putText(frame, f"Exercise: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if exercise:
        cv2.putText(
            frame,
            "Auto-detected",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (100, 255, 100),
            1,
        )

        cv2.rectangle(frame, (w - 170, 38), (w - 15, 98), (15, 15, 15), -1)
        cv2.putText(frame, "REPS", (w - 160, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255), 1)
        cv2.putText(frame, str(reps), (w - 95, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 120), 2)

    cv2.putText(frame, "Q = quit", (w - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if exercise:
        active_feedback = feedback[:2] if feedback else ["Feedback active - keep controlled full range"]
        box_height = 24 + len(active_feedback) * 34

        # Keep feedback comfortably above lower-body occlusion zones.
        bottom_offset = 220
        if exercise == "squat":
            bottom_offset = 280
        elif exercise == "curl":
            bottom_offset = 240

        box_top = max(90, h - box_height - bottom_offset)
        box_bottom = min(h - 24, box_top + box_height)
        cv2.rectangle(frame, (0, box_top), (w, box_bottom), (0, 0, 0), -1)
        for i, line in enumerate(active_feedback):
            y = box_top + 28 + i * 32
            if "not counted" in line.lower():
                line_color = (0, 80, 255)
            elif "good" in line.lower() or "great" in line.lower() or "counted" in line.lower():
                line_color = (0, 255, 100)
            else:
                line_color = (0, 130, 255)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, line_color, 2)


def default_live_feedback(exercise, tracker):
    if not exercise or tracker is None:
        return []

    if exercise == "squat":
        if tracker.phase == "down":
            return ["Lower hips under control - knees track over toes"]
        return ["Drive through heels and stand tall"]

    if exercise == "curl":
        if tracker.phase == "down":
            return ["Curl up toward shoulder - keep elbows tucked"]
        return ["Lower slowly until arms are almost straight"]

    if exercise == "pushup":
        if tracker.phase == "down":
            return ["Lower chest under control - keep core tight"]
        return ["Press up while keeping body in one line"]

    if exercise == "incline_chest_press":
        if tracker.phase == "down":
            return ["Lower until dumbbells are close to chest level"]
        return ["Press up, then lower with control"]

    return ["Keep steady form and full range of motion"]


def is_start_position(exercise, angles, landmarks=None):
    if not angles or not exercise:
        return False

    avg_knee = (angles["left_knee"] + angles["right_knee"]) / 2.0
    avg_hip = (angles["left_hip"] + angles["right_hip"]) / 2.0
    avg_elbow = (angles["left_elbow"] + angles["right_elbow"]) / 2.0

    if exercise == "squat":
        return avg_knee > 150 and avg_hip > 145
    if exercise == "curl":
        return avg_elbow > 145
    if exercise == "pushup":
        return avg_elbow > 150 and avg_hip > 155
    if exercise == "incline_chest_press":
        if landmarks is None:
            return False

        lm = landmarks.landmark
        left_wrist = lm[L.LEFT_WRIST.value]
        right_wrist = lm[L.RIGHT_WRIST.value]
        nose = lm[L.NOSE.value]

        # Counter starts only when dumbbells are above head level.
        avg_wrist_y = (left_wrist.y + right_wrist.y) / 2.0
        wrists_above_head = avg_wrist_y < (nose.y + 0.035)
        return avg_elbow > 135 and wrists_above_head
    return True


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print("Cannot open source:", src)
        return

    trackers = {ex: make_tracker(ex) for ex in EXERCISES}
    last_feedback = []
    sticky_feedback = []
    sticky_timer = 0

    stable_exercise = None
    active_exercise = None
    candidate_exercise = None
    candidate_frames = 0
    missing_frames = 0
    last_exercise = None

    detection_lock_frames = 6
    detection_switch_frames = 30
    detection_release_frames = 75
    active_switch_frames = 240
    active_release_frames = 300
    tracking_warmup_frames = 2
    feedback_hold_frames = 180
    stable_frames = 0
    active_candidate = None
    active_candidate_frames = 0
    reps_memory = {ex: 0 for ex in EXERCISES}
    displayed_reps = 0
    displayed_exercise = None
    counting_enabled = False
    start_gate_frames = 24
    ready_visible_frames = 0

    print("Running - Q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (540, 960))
        landmarks = get_landmarks(frame)
        angles = None

        if landmarks:
            draw_skeleton(frame, landmarks)
            angles = extract_angles(landmarks)
            guess = detect_exercise(angles, landmarks)

            if guess:
                missing_frames = 0
                if guess == stable_exercise:
                    candidate_exercise = None
                    candidate_frames = 0
                else:
                    if guess == candidate_exercise:
                        candidate_frames += 1
                    else:
                        candidate_exercise = guess
                        candidate_frames = 1

                    needed = detection_lock_frames if stable_exercise is None else detection_switch_frames
                    if candidate_frames >= needed:
                        stable_exercise = guess
                        candidate_exercise = None
                        candidate_frames = 0
            else:
                missing_frames += 1
                if missing_frames > detection_release_frames:
                    stable_exercise = None
                    candidate_exercise = None
                    candidate_frames = 0
        else:
            missing_frames += 1
            if missing_frames > detection_release_frames:
                stable_exercise = None
                candidate_exercise = None
                candidate_frames = 0

        if stable_exercise != last_exercise:
            last_feedback = []
            stable_frames = 0
            last_exercise = stable_exercise
        elif stable_exercise:
            stable_frames += 1

        # Lock the counting/display exercise to prevent rapid tracker swapping.
        if active_exercise is None:
            if stable_exercise:
                active_exercise = stable_exercise
                active_candidate = None
                active_candidate_frames = 0
                trackers[active_exercise] = make_tracker(active_exercise)
                reps_memory[active_exercise] = 0
                displayed_reps = 0
                displayed_exercise = active_exercise
                counting_enabled = False
                ready_visible_frames = 0
        else:
            if stable_exercise == active_exercise:
                active_candidate = None
                active_candidate_frames = 0
            elif stable_exercise:
                if stable_exercise == active_candidate:
                    active_candidate_frames += 1
                else:
                    active_candidate = stable_exercise
                    active_candidate_frames = 1
                if active_candidate_frames >= active_switch_frames:
                    active_exercise = active_candidate
                    active_candidate = None
                    active_candidate_frames = 0
                    last_feedback = []
                    sticky_feedback = []
                    sticky_timer = 0
                    trackers[active_exercise] = make_tracker(active_exercise)
                    reps_memory[active_exercise] = 0
                    displayed_reps = 0
                    displayed_exercise = active_exercise
                    counting_enabled = False
                    ready_visible_frames = 0
            elif missing_frames > detection_release_frames:
                # Keep active exercise sticky through short pose dropouts.
                if missing_frames > active_release_frames:
                    active_exercise = None
                    active_candidate = None
                    active_candidate_frames = 0
                    last_feedback = []
                    sticky_feedback = []
                    sticky_timer = 0
                    counting_enabled = False
                    ready_visible_frames = 0

        reps = trackers[active_exercise].reps if active_exercise else 0
        if active_exercise != displayed_exercise:
            displayed_exercise = active_exercise
            displayed_reps = reps if active_exercise else 0

        if active_exercise:
            last_feedback = default_live_feedback(active_exercise, trackers[active_exercise])
        else:
            last_feedback = []
            sticky_feedback = []
            sticky_timer = 0

        visible_for_active = (
            landmarks
            and angles
            and active_exercise
            and stable_frames >= tracking_warmup_frames
            and has_required_visibility(landmarks, active_exercise)
        )

        if active_exercise and not counting_enabled:
            if visible_for_active and is_start_position(active_exercise, angles, landmarks):
                ready_visible_frames += 1
            else:
                ready_visible_frames = 0
                trackers[active_exercise] = make_tracker(active_exercise)

            gate_needed = 8 if active_exercise == "incline_chest_press" else start_gate_frames
            if ready_visible_frames >= gate_needed:
                # Start counting only after user has held a proper start posture.
                counting_enabled = True
                trackers[active_exercise] = make_tracker(active_exercise)
                reps_memory[active_exercise] = 0
                displayed_reps = 0
                last_feedback = ["Start reps now - counter is active"]
                sticky_feedback = ["Start reps now - counter is active"]
                sticky_timer = 60
            else:
                last_feedback = ["Get set - hold start position to begin counting"]
                sticky_feedback = []
                sticky_timer = 0
                reps = 0

        elif visible_for_active:
            new_feedback = check_form(active_exercise, angles, trackers[active_exercise])
            t = trackers[active_exercise]
            print(f"phase:{t.phase} smooth:{sum(t.buffer)/max(len(t.buffer),1):.0f} min:{t.min_angle:.0f} reps:{t.reps}")
            reps = trackers[active_exercise].reps

            if new_feedback:
                sticky_feedback = new_feedback[:4]
                sticky_timer = feedback_hold_frames

        if sticky_timer > 0 and sticky_feedback:
            last_feedback = sticky_feedback
            sticky_timer -= 1

        if active_exercise and counting_enabled:
            reps_memory[active_exercise] = max(reps_memory[active_exercise], reps)
            displayed_reps = max(displayed_reps, reps_memory[active_exercise])
        elif active_exercise and not counting_enabled:
            displayed_reps = 0
        render_exercise = active_exercise if active_exercise else displayed_exercise
        draw_feedback(frame, last_feedback, render_exercise, displayed_reps)
        cv2.imshow("Gym Form Checker", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
