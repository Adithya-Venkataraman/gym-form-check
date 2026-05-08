import mediapipe as mp

L = mp.solutions.pose.PoseLandmark


class RepTracker:
    def __init__(
        self,
        down_thresh,
        up_thresh,
        min_rep_range=12.0,
        buffer_size=3,
        transition_frames=1,
        min_frames_between_reps=10,
    ):
        self.down_thresh = down_thresh
        self.up_thresh = up_thresh
        self.min_rep_range = min_rep_range
        self.buffer_size = buffer_size
        self.transition_frames = transition_frames
        self.min_frames_between_reps = min_frames_between_reps

        self.phase = "up"
        self.reps = 0
        self.min_angle = 999
        self.max_angle = 0
        self.buffer = []
        self.down_streak = 0
        self.up_streak = 0
        self.frames_since_rep = 10000

        self.feedback_due = False
        self.last_cycle_min = None
        self.last_cycle_range = 0
        self.last_cycle_counted = False

    def update(self, angle):
        self.buffer.append(angle)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        smooth = sum(self.buffer) / len(self.buffer)
        self.frames_since_rep += 1

        self.min_angle = min(self.min_angle, smooth)
        self.max_angle = max(self.max_angle, smooth)

        if len(self.buffer) < self.buffer_size:
            return

        if self.phase == "up":
            if smooth < self.down_thresh:
                self.down_streak += 1
            else:
                self.down_streak = 0

            if self.down_streak >= self.transition_frames:
                self.phase = "down"
                self.down_streak = 0
                self.min_angle = smooth
                self.max_angle = smooth
            return

        if self.phase == "down" and smooth > self.up_thresh:
            self.up_streak += 1
            if self.up_streak >= self.transition_frames:
                cycle_range = self.max_angle - self.min_angle
                rep_counted = (
                    cycle_range >= self.min_rep_range
                    and self.frames_since_rep >= self.min_frames_between_reps
                )
                if rep_counted:
                    self.reps += 1
                    self.frames_since_rep = 0

                self.last_cycle_min = self.min_angle
                self.last_cycle_range = cycle_range
                self.last_cycle_counted = rep_counted
                self.feedback_due = cycle_range >= self.min_rep_range * 0.6
                self.phase = "up"
                self.min_angle = 999
                self.max_angle = 0
                self.up_streak = 0
        else:
            self.up_streak = 0

    def pop_feedback_context(self):
        if not self.feedback_due:
            return None
        self.feedback_due = False
        return {
            "min_angle": self.last_cycle_min,
            "range": self.last_cycle_range,
            "counted": self.last_cycle_counted,
        }


def make_tracker(exercise):
    configs = {
        "squat": RepTracker(
            down_thresh=118,
            up_thresh=152,
            min_rep_range=14,
            transition_frames=2,
            min_frames_between_reps=12,
        ),
        "pushup": RepTracker(
            down_thresh=108,
            up_thresh=150,
            min_rep_range=12,
            transition_frames=2,
            min_frames_between_reps=10,
        ),
        "curl": RepTracker(
            down_thresh=100,
            up_thresh=138,
            min_rep_range=12,
            transition_frames=2,
            min_frames_between_reps=12,
        ),
        "incline_chest_press": RepTracker(
            down_thresh=118,
            up_thresh=148,
            min_rep_range=12,
            transition_frames=2,
            min_frames_between_reps=12,
        ),
    }
    return configs.get(exercise, RepTracker(90, 150))


ANGLE_JOINTS = {
    "knee": (
        [L.LEFT_HIP.value, L.LEFT_KNEE.value, L.LEFT_ANKLE.value],
        [L.RIGHT_HIP.value, L.RIGHT_KNEE.value, L.RIGHT_ANKLE.value],
    ),
    "hip": (
        [L.LEFT_SHOULDER.value, L.LEFT_HIP.value, L.LEFT_KNEE.value],
        [L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value, L.RIGHT_KNEE.value],
    ),
    "elbow": (
        [L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value],
        [L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value],
    ),
    "shoulder": (
        [L.LEFT_ELBOW.value, L.LEFT_SHOULDER.value, L.LEFT_HIP.value],
        [L.RIGHT_ELBOW.value, L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value],
    ),
}


def _min_visibility(landmarks, indices):
    lm = landmarks.landmark
    return min(lm[i].visibility for i in indices)


def _best_angle(angles, landmarks, name):
    if landmarks is None:
        left_key = f"left_{name}"
        right_key = f"right_{name}"
        return min(angles[left_key], angles[right_key]), "left", 1.0

    left_indices, right_indices = ANGLE_JOINTS[name]
    left_vis = _min_visibility(landmarks, left_indices)
    right_vis = _min_visibility(landmarks, right_indices)
    side = "left" if left_vis >= right_vis else "right"
    key = f"{side}_{name}"
    vis = left_vis if side == "left" else right_vis
    return angles[key], side, vis


def _blended_angle(angles, landmarks, name):
    left_key = f"left_{name}"
    right_key = f"right_{name}"
    if landmarks is None:
        return (angles[left_key] + angles[right_key]) / 2.0

    left_indices, right_indices = ANGLE_JOINTS[name]
    left_vis = max(0.01, _min_visibility(landmarks, left_indices))
    right_vis = max(0.01, _min_visibility(landmarks, right_indices))
    total = left_vis + right_vis
    return (angles[left_key] * left_vis + angles[right_key] * right_vis) / total


def _point(lm, side, joint):
    if side == "left":
        index = getattr(L, f"LEFT_{joint}").value
    else:
        index = getattr(L, f"RIGHT_{joint}").value
    return lm[index]


def _update_squat_rep_by_hip(tracker, hip_y):
    # Lazy-init squat-specific state on the shared tracker object.
    if not hasattr(tracker, "hip_buffer"):
        tracker.hip_buffer = []
        tracker.squat_phase = "up"
        tracker.top_hip = hip_y
        tracker.bottom_hip = hip_y
        tracker.down_streak = 0
        tracker.up_streak = 0
        tracker.frames_since_rep = 10000
        tracker.last_cycle_counted = False
        tracker.last_cycle_range = 0.0
        tracker.last_cycle_min = 0.0
        tracker.feedback_due = False

    tracker.frames_since_rep += 1
    tracker.hip_buffer.append(hip_y)
    if len(tracker.hip_buffer) > 5:
        tracker.hip_buffer.pop(0)
    smooth_hip = sum(tracker.hip_buffer) / len(tracker.hip_buffer)

    down_delta_start = 0.035
    up_delta_finish = 0.035
    full_cycle_delta = 0.055
    min_frames_between_reps = 10
    transition_frames = 2

    if tracker.squat_phase == "up":
        tracker.top_hip = min(tracker.top_hip, smooth_hip)
        if smooth_hip - tracker.top_hip > down_delta_start:
            tracker.down_streak += 1
        else:
            tracker.down_streak = 0

        if tracker.down_streak >= transition_frames:
            tracker.squat_phase = "down"
            tracker.bottom_hip = smooth_hip
            tracker.down_streak = 0
        return

    tracker.bottom_hip = max(tracker.bottom_hip, smooth_hip)
    if tracker.bottom_hip - smooth_hip > up_delta_finish:
        tracker.up_streak += 1
    else:
        tracker.up_streak = 0

    if tracker.up_streak >= transition_frames:
        cycle_range = tracker.bottom_hip - tracker.top_hip
        rep_counted = cycle_range >= full_cycle_delta and tracker.frames_since_rep >= min_frames_between_reps

        if rep_counted:
            tracker.reps += 1
            tracker.frames_since_rep = 0

        tracker.last_cycle_counted = rep_counted
        tracker.last_cycle_range = cycle_range
        tracker.last_cycle_min = tracker.top_hip
        tracker.feedback_due = True

        tracker.squat_phase = "up"
        tracker.top_hip = smooth_hip
        tracker.bottom_hip = smooth_hip
        tracker.up_streak = 0


def _update_incline_rep_by_wrist(tracker, wrist_y, shoulder_y, torso_span, elbow_angle):
    # Simple and stable FSM:
    # 1) detect top/lockout
    # 2) detect clear descent
    # 3) detect clear ascent back to top region => +1 rep
    if not hasattr(tracker, "press_buffer"):
        tracker.press_buffer = []
        tracker.elbow_buffer = []
        tracker.phase = "up"
        tracker.press_state = "wait_top"
        tracker.top_depth_ref = 0.0
        tracker.bottom_depth = 0.0
        tracker.bottom_elbow = elbow_angle
        tracker.top_streak = 0
        tracker.down_streak = 0
        tracker.up_streak = 0
        tracker.cooldown = 0
        tracker.frames_since_rep = 10000
        tracker.last_cycle_counted = False
        tracker.last_cycle_range = 0.0
        tracker.last_cycle_min = 0.0
        tracker.feedback_due = False
        tracker.initialized = False

    tracker.frames_since_rep += 1
    if tracker.cooldown > 0:
        tracker.cooldown -= 1

    tracker.press_buffer.append(wrist_y)
    tracker.elbow_buffer.append(elbow_angle)
    if len(tracker.press_buffer) > 5:
        tracker.press_buffer.pop(0)
    if len(tracker.elbow_buffer) > 5:
        tracker.elbow_buffer.pop(0)

    smooth_wrist = sum(tracker.press_buffer) / len(tracker.press_buffer)
    smooth_elbow = sum(tracker.elbow_buffer) / len(tracker.elbow_buffer)
    depth = smooth_wrist - shoulder_y

    top_depth_limit = max(0.030, torso_span * 0.14)
    down_depth_needed = max(0.050, torso_span * 0.22)
    up_depth_needed = max(0.030, torso_span * 0.14)
    top_elbow_thresh = 145.0
    bottom_elbow_thresh = 118.0
    up_elbow_thresh = 136.0
    transition_frames = 2
    min_frames_between_reps = 14

    if not tracker.initialized:
        tracker.top_depth_ref = depth
        tracker.bottom_depth = depth
        tracker.bottom_elbow = smooth_elbow
        tracker.initialized = True
        return

    # Wait until lockout/top is clearly reached.
    if tracker.press_state == "wait_top":
        tracker.phase = "up"
        near_top = depth <= top_depth_limit and smooth_elbow >= top_elbow_thresh
        if near_top:
            tracker.top_streak += 1
            tracker.top_depth_ref = min(tracker.top_depth_ref, depth)
        else:
            tracker.top_streak = max(0, tracker.top_streak - 1)
            tracker.top_depth_ref = min(tracker.top_depth_ref * 0.98 + depth * 0.02, depth)

        if tracker.top_streak >= transition_frames:
            tracker.press_state = "go_down"
            tracker.top_streak = 0
            tracker.bottom_depth = depth
            tracker.bottom_elbow = smooth_elbow
        return

    # Wait for real descent toward chest.
    if tracker.press_state == "go_down":
        tracker.phase = "down"
        tracker.bottom_depth = max(tracker.bottom_depth, depth)
        tracker.bottom_elbow = min(tracker.bottom_elbow, smooth_elbow)

        descended_enough = (
            (tracker.bottom_depth - tracker.top_depth_ref) >= down_depth_needed
            and tracker.bottom_elbow <= bottom_elbow_thresh
        )
        if descended_enough:
            tracker.down_streak += 1
        else:
            tracker.down_streak = 0

        if tracker.down_streak >= transition_frames:
            tracker.press_state = "go_up"
            tracker.down_streak = 0
            tracker.up_streak = 0
        return

    # Wait for clear ascent back toward top zone.
    tracker.phase = "up"
    risen_enough = (tracker.bottom_depth - depth) >= up_depth_needed
    elbows_extended = smooth_elbow >= up_elbow_thresh
    back_to_top_zone = depth <= (top_depth_limit + 0.010)

    if risen_enough and elbows_extended and back_to_top_zone:
        tracker.up_streak += 1
    else:
        tracker.up_streak = 0

    if tracker.up_streak >= transition_frames:
        cycle_range = tracker.bottom_depth - tracker.top_depth_ref
        rep_counted = (
            cycle_range >= down_depth_needed
            and tracker.frames_since_rep >= min_frames_between_reps
            and tracker.cooldown == 0
        )

        if rep_counted:
            tracker.reps += 1
            tracker.frames_since_rep = 0
            tracker.cooldown = 8

        tracker.last_cycle_counted = rep_counted
        tracker.last_cycle_range = cycle_range
        tracker.last_cycle_min = tracker.top_depth_ref
        tracker.feedback_due = rep_counted or cycle_range >= down_depth_needed * 0.8

        tracker.press_state = "wait_top"
        tracker.top_depth_ref = depth
        tracker.bottom_depth = depth
        tracker.bottom_elbow = smooth_elbow
        tracker.up_streak = 0


def check_squat(angles, tracker, landmarks=None):
    feedback = []
    knee = _blended_angle(angles, landmarks, "knee")
    hip = _blended_angle(angles, landmarks, "hip")
    _, side, _ = _best_angle(angles, landmarks, "knee")
    cycle = None

    if landmarks is not None:
        lm = landmarks.landmark
        hip_pt = _point(lm, side, "HIP")
        _update_squat_rep_by_hip(tracker, hip_pt.y)
        cycle = tracker.pop_feedback_context()
    else:
        tracker.update(knee)
        cycle = tracker.pop_feedback_context()

    if cycle:
        min_knee = cycle["min_angle"]
        if cycle["counted"]:
            if min_knee > 118:
                feedback.append("Rep counted - good rep. Try going a bit deeper")
            elif min_knee > 102:
                feedback.append("Rep counted - nice rep. Slightly deeper for better range")
            elif min_knee < 60:
                feedback.append("Rep counted - great depth, keep control")
            else:
                feedback.append("Rep counted - good form")
        else:
            feedback.append("No rep - move through a clearer down-up range")

    if hip < 60:
        feedback.append("Keep chest up - avoid folding forward")

    if landmarks is not None and tracker.phase == "down":
        lm = landmarks.landmark
        hip_pt = _point(lm, side, "HIP")
        knee_pt = _point(lm, side, "KNEE")
        if hip_pt.y < knee_pt.y - 0.05:
            feedback.append("Sit back and down - hips should reach knee level")

    return feedback


def check_pushup(angles, tracker, landmarks=None):
    feedback = []
    elbow = _blended_angle(angles, landmarks, "elbow")
    hip = _blended_angle(angles, landmarks, "hip")

    tracker.update(elbow)
    cycle = tracker.pop_feedback_context()

    if cycle:
        min_elbow = cycle["min_angle"]
        if cycle["counted"]:
            if min_elbow > 100:
                feedback.append("Rep counted - good rep. Go a bit lower next time")
            elif min_elbow < 60:
                feedback.append("Rep counted - good effort. Do not drop too low")
            else:
                feedback.append("Rep counted - good pushup rep")
        else:
            feedback.append("No rep - complete a full down-up pushup")

    if hip < 155:
        feedback.append("Hips sagging - tighten your core")
    elif hip > 210:
        feedback.append("Hips too high - lower them down")

    return feedback


def check_curl(angles, tracker, landmarks=None):
    feedback = []
    elbow = _blended_angle(angles, landmarks, "elbow")
    shoulder = _blended_angle(angles, landmarks, "shoulder")

    tracker.update(elbow)
    cycle = tracker.pop_feedback_context()

    if cycle:
        if cycle["counted"]:
            if cycle["min_angle"] > 95:
                feedback.append("Rep counted - good rep. Curl slightly higher")
            else:
                feedback.append("Rep counted - good curl rep")
        else:
            feedback.append("No rep - lower then curl through fuller range")

    if shoulder > 50:
        feedback.append("Elbows drifting - keep them tucked")

    return feedback


def check_incline_chest_press(angles, tracker, landmarks=None):
    feedback = []
    elbow = _blended_angle(angles, landmarks, "elbow")
    shoulder = _blended_angle(angles, landmarks, "shoulder")
    hip_angle = _blended_angle(angles, landmarks, "hip")

    if landmarks is not None:
        torso_left_vis = _min_visibility(landmarks, [L.LEFT_SHOULDER.value, L.LEFT_HIP.value])
        torso_right_vis = _min_visibility(landmarks, [L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value])
        torso_side = "left" if torso_left_vis >= torso_right_vis else "right"
        lm = landmarks.landmark
        shoulder_pt = _point(lm, torso_side, "SHOULDER")
        hip_pt = _point(lm, torso_side, "HIP")
        torso_delta = abs(shoulder_pt.y - hip_pt.y)
        torso_span = max(0.08, abs(hip_pt.y - shoulder_pt.y))

        if torso_delta > 0.20 and hip_angle > 145:
            return ["Set up on an incline bench to start chest press"]

        left_vis = _min_visibility(landmarks, [L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value])
        right_vis = _min_visibility(landmarks, [L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value])
        left_wrist_y = lm[L.LEFT_WRIST.value].y
        right_wrist_y = lm[L.RIGHT_WRIST.value].y

        if left_vis > 0.25 and right_vis > 0.25:
            avg_wrist_y = (left_wrist_y + right_wrist_y) / 2.0
        elif left_vis >= right_vis:
            avg_wrist_y = left_wrist_y
        else:
            avg_wrist_y = right_wrist_y

        _update_incline_rep_by_wrist(tracker, avg_wrist_y, shoulder_pt.y, torso_span, elbow)
    else:
        tracker.update(elbow)

    cycle = tracker.pop_feedback_context()

    if cycle:
        min_elbow = cycle["min_angle"]
        if cycle["counted"]:
            if min_elbow > 112:
                feedback.append("Rep counted - good rep. Lower a little deeper")
            elif min_elbow > 95:
                feedback.append("Rep counted - good rep. Slightly more depth")
            elif min_elbow < 55:
                feedback.append("Rep counted - good rep. Avoid going too deep")
            else:
                feedback.append("Rep counted - good chest press rep")
        else:
            feedback.append("No rep - complete a clear down-up press")

    if tracker.phase == "down" and elbow > 112:
        feedback.append("Keep lowering under control")
    if shoulder > 120:
        feedback.append("Tuck elbows slightly - keep them around 45 degrees")
    elif shoulder < 25:
        feedback.append("Open elbows slightly - do not over-tuck")
    if not feedback:
        if tracker.phase == "down":
            feedback.append("Lower until dumbbells are close to chest level")
        else:
            feedback.append("Press up, then lower with control")

    return feedback


EXERCISES = {
    "squat": check_squat,
    "pushup": check_pushup,
    "curl": check_curl,
    "incline_chest_press": check_incline_chest_press,
}


def detect_exercise(angles, landmarks):
    lm = landmarks.landmark

    knee_angle, leg_side, leg_vis = _best_angle(angles, landmarks, "knee")
    hip_angle, _, hip_vis = _best_angle(angles, landmarks, "hip")
    elbow_angle, arm_side, arm_vis = _best_angle(angles, landmarks, "elbow")
    shoulder_angle, _, _ = _best_angle(angles, landmarks, "shoulder")

    torso_left_vis = _min_visibility(landmarks, [L.LEFT_SHOULDER.value, L.LEFT_HIP.value])
    torso_right_vis = _min_visibility(landmarks, [L.RIGHT_SHOULDER.value, L.RIGHT_HIP.value])
    torso_side = "left" if torso_left_vis >= torso_right_vis else "right"

    shoulder = _point(lm, torso_side, "SHOULDER")
    hip = _point(lm, torso_side, "HIP")
    knee = _point(lm, leg_side, "KNEE")
    ankle = _point(lm, leg_side, "ANKLE")
    wrist = _point(lm, arm_side, "WRIST")
    elbow = _point(lm, arm_side, "ELBOW")

    torso_delta = abs(shoulder.y - hip.y)
    upright_body = shoulder.y < hip.y - 0.09 and torso_delta > 0.16
    reclined_body = torso_delta < 0.18
    standing_like = upright_body and knee_angle > 160 and hip_angle > 150
    seated_or_folded = knee_angle < 152 or hip_angle < 148

    scores = {
        "squat": 0.0,
        "incline_chest_press": 0.0,
        "curl": 0.0,
        "pushup": 0.0,
    }

    if min(leg_vis, hip_vis) > 0.35:
        scores["squat"] += 0.15
    if upright_body:
        scores["squat"] += 0.30
    if knee_angle < 155:
        scores["squat"] += 0.30
    if hip.y > 0.42:
        scores["squat"] += 0.10
    if ankle.y > knee.y > hip.y - 0.02:
        scores["squat"] += 0.15
    if knee_angle < 120:
        scores["squat"] += 0.10
    if reclined_body:
        scores["squat"] -= 0.20

    if min(arm_vis, hip_vis) > 0.35:
        scores["incline_chest_press"] += 0.15
    if reclined_body and seated_or_folded:
        scores["incline_chest_press"] += 0.35
    if hip_angle < 145:
        scores["incline_chest_press"] += 0.12
    if elbow_angle < 165:
        scores["incline_chest_press"] += 0.20
    if wrist.y < shoulder.y + 0.20:
        scores["incline_chest_press"] += 0.15
    if elbow.y < hip.y + 0.16:
        scores["incline_chest_press"] += 0.10
    if upright_body:
        scores["incline_chest_press"] -= 0.30
    if standing_like:
        scores["incline_chest_press"] -= 0.55
    if not seated_or_folded:
        scores["incline_chest_press"] -= 0.20
    if hip_angle > 155:
        scores["incline_chest_press"] -= 0.35

    if upright_body:
        scores["curl"] += 0.35
    if elbow_angle < 132:
        scores["curl"] += 0.30
    if shoulder_angle < 75:
        scores["curl"] += 0.20
    if wrist.y < hip.y + 0.12:
        scores["curl"] += 0.10
    if knee_angle > 155:
        scores["curl"] += 0.10
    if knee_angle < 148:
        scores["curl"] -= 0.30
    if reclined_body:
        scores["curl"] -= 0.25

    if reclined_body:
        scores["pushup"] += 0.25
    if elbow_angle < 165:
        scores["pushup"] += 0.20
    if hip.y > 0.62:
        scores["pushup"] += 0.20
    if wrist.y > shoulder.y - 0.05:
        scores["pushup"] += 0.10
    if knee_angle > 145:
        scores["pushup"] += 0.15
    if upright_body:
        scores["pushup"] -= 0.20

    if hip_angle < 175:
        scores["squat"] += 0.03
        scores["pushup"] += 0.02

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_exercise, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_score < 0.62:
        return None
    if best_score - second_score < 0.08:
        return None

    return best_exercise


def check_form(exercise, angles, tracker, landmarks=None):
    checker = EXERCISES.get(exercise)
    if checker is None:
        return [f"Unknown exercise: {exercise}"]
    return checker(angles, tracker, landmarks)
