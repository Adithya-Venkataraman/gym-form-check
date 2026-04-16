import mediapipe as mp
L = mp.solutions.pose.PoseLandmark


class RepTracker:
    def __init__(self, down_thresh, up_thresh):
        self.down_thresh  = down_thresh
        self.up_thresh    = up_thresh
        self.phase        = "up"
        self.reps         = 0
        self.min_angle    = 999
        self.max_angle    = 0
        self.feedback_due = False
        self.buffer       = []
        self.buffer_size  = 5

    def update(self, angle):
        self.buffer.append(angle)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        smooth = sum(self.buffer) / len(self.buffer)

        self.min_angle = min(self.min_angle, smooth)
        self.max_angle = max(self.max_angle, smooth)

        # wait for full buffer before tracking
        if len(self.buffer) < self.buffer_size:
            self.phase = "down" if smooth < self.down_thresh else "up"
            return

        if self.phase == "up" and smooth < self.down_thresh:
            self.phase        = "down"
            self.feedback_due = True

        elif self.phase == "down" and smooth > self.up_thresh:
            self.phase     = "up"
            self.reps     += 1
            self.min_angle = 999
            self.max_angle = 0

    def pop_feedback_due(self):
        if self.feedback_due:
            self.feedback_due = False
            return True
        return False


def make_tracker(exercise):
    configs = {
        "squat":               RepTracker(down_thresh=100, up_thresh=155),
        "pushup":              RepTracker(down_thresh=90,  up_thresh=155),
        "curl":                RepTracker(down_thresh=60,  up_thresh=140),
        "incline_chest_press": RepTracker(down_thresh=85,  up_thresh=108),
    }
    return configs.get(exercise, RepTracker(90, 150))

def check_squat(angles, tracker):
    feedback = []
    knee = min(angles["left_knee"], angles["right_knee"])
    hip  = min(angles["left_hip"],  angles["right_hip"])

    tracker.update(knee)

    if tracker.pop_feedback_due():
        if tracker.min_angle > 110:
            feedback.append("Bend your knees more — go deeper")
        elif tracker.min_angle > 95:
            feedback.append("Almost there — just a bit deeper")
        elif tracker.min_angle < 50:
            feedback.append("Too deep — stop at parallel")
        else:
            feedback.append("Perfect depth!")

    # continuous checks every frame
    if hip > 130:
        feedback.append("Chest too low — don't lean forward")
    elif hip < 55:
        feedback.append("Hinge forward slightly at the hips")

    return feedback


def check_pushup(angles, tracker):
    feedback = []
    elbow = min(angles["left_elbow"], angles["right_elbow"])
    hip   = min(angles["left_hip"],   angles["right_hip"])

    tracker.update(elbow)

    if tracker.pop_feedback_due():
        if tracker.min_angle > 100:
            feedback.append("Go lower — chest not reaching bottom")
        elif tracker.min_angle < 60:
            feedback.append("Too low — stop at 90°")
        else:
            feedback.append("Good depth!")

    if hip < 155:
        feedback.append("Hips sagging — tighten your core")
    elif hip > 210:
        feedback.append("Hips too high — lower them down")

    return feedback


def check_curl(angles, tracker):
    feedback = []
    elbow    = min(angles["left_elbow"],    angles["right_elbow"])
    shoulder = min(angles["left_shoulder"], angles["right_shoulder"])

    tracker.update(elbow)

    if tracker.pop_feedback_due():
        if tracker.min_angle > 80:
            feedback.append("Curl higher — bring weight to shoulder")
        else:
            feedback.append("Good curl depth!")

    if shoulder > 50:
        feedback.append("Elbows drifting — keep them tucked")

    return feedback


def check_incline_chest_press(angles, tracker):
    feedback = []
    elbow = min(angles["left_elbow"], angles["right_elbow"])

    tracker.update(elbow)

    if tracker.pop_feedback_due():
        if tracker.min_angle > 90:
            feedback.append("Lower your arms more — bring dumbbells to chest level")
        elif tracker.min_angle < 70:
            feedback.append("Too low — stop when elbows reach shoulder level")
        else:
            feedback.append("Great range of motion — keep it up!")

    return feedback


EXERCISES = {
    "squat":               check_squat,
    "pushup":              check_pushup,
    "curl":                check_curl,
    "incline_chest_press": check_incline_chest_press,
}


def detect_exercise(angles, landmarks):
    lm = landmarks.landmark

    hip_y      = (lm[L.LEFT_HIP].y      + lm[L.RIGHT_HIP].y)      / 2
    shoulder_y = (lm[L.LEFT_SHOULDER].y + lm[L.RIGHT_SHOULDER].y) / 2
    wrist_y    = (lm[L.LEFT_WRIST].y    + lm[L.RIGHT_WRIST].y)    / 2
    knee_y     = (lm[L.LEFT_KNEE].y     + lm[L.RIGHT_KNEE].y)     / 2

    avg_knee     = (angles["left_knee"]     + angles["right_knee"])     / 2
    avg_elbow    = (angles["left_elbow"]    + angles["right_elbow"])    / 2
    avg_shoulder = (angles["left_shoulder"] + angles["right_shoulder"]) / 2

    arms_raised   = wrist_y < shoulder_y
    body_upright  = shoulder_y < hip_y - 0.05
    # person is lying down — shoulder and hip at similar height
    body_lying    = abs(shoulder_y - hip_y) < 0.2

    # squat — upright body, knees bent, feet on ground
    if body_upright and avg_knee < 150 and hip_y > 0.5:
        return "squat"

    # incline chest press — only when body is lying/reclined AND arms raised
    if body_lying and arms_raised and avg_elbow < 160:
        return "incline_chest_press"

    # curl — upright, elbow bent, minimal shoulder movement
    if body_upright and avg_elbow < 130 and avg_shoulder < 60:
        return "curl"

    # pushup — low hip position
    if hip_y > 0.75 and avg_elbow < 160:
        return "pushup"

    return None


def check_form(exercise, angles, tracker):
    checker = EXERCISES.get(exercise)
    if checker is None:
        return [f"Unknown exercise: {exercise}"]
    return checker(angles, tracker)