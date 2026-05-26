from dataclasses import dataclass


ACTIONS = [
    "SA-12_WRIST_STRAIGHTEN",
    "SA-13_ELBOW_RAISE",
    "SA-14_FINGER_CURVE",
    "SA-15_SHOULDER_RELAX",
    "SA-16_THUMB_RELAX",
    "SA-17_POSITIVE_T",
    "SA-18_SWITCH_POSTURE_REPEAT",
]


FEATURE_TO_STATE = {
    0: "LEFT_WRIST",
    1: "ELBOW_WIDTH",
    2: "WRIST_SHAKE",
    3: "LEFT_ARM",
    4: "RIGHT_ARM",
    5: "RIGHT_WRIST",
}

STATE_TO_ACTION = {
    "LEFT_WRIST": "SA-12_WRIST_STRAIGHTEN",
    "RIGHT_WRIST": "SA-12_WRIST_STRAIGHTEN",
    "WRIST_SHAKE": "SA-12_WRIST_STRAIGHTEN",
    "ELBOW_WIDTH": "SA-13_ELBOW_RAISE",
    "LEFT_ARM": "SA-13_ELBOW_RAISE",
    "RIGHT_ARM": "SA-13_ELBOW_RAISE",
    "FINGER_CURVED": "SA-14_FINGER_CURVE",
    "SHOULDER": "SA-15_SHOULDER_RELAX",
    "THUMB_TENSION": "SA-16_THUMB_RELAX",
    "GOOD": "SA-17_POSITIVE_T",
}


ACTION_FEEDBACK = {
    "SA-12_WRIST_STRAIGHTEN": "손목을 곧게 펴세요.",
    "SA-13_ELBOW_RAISE": "팔꿈치 높이와 팔 각도를 안정적으로 맞추세요.",
    "SA-14_FINGER_CURVE": "손가락을 둥글게 세우세요.",
    "SA-15_SHOULDER_RELAX": "어깨에 힘을 빼고 편안하게 낮추세요.",
    "SA-16_THUMB_RELAX": "엄지에 들어간 힘을 풀어주세요.",
    "SA-17_POSITIVE_T": "좋아요. 현재 자세를 유지하세요.",
    "SA-18_SWITCH_POSTURE_REPEAT": "같은 문제가 반복되고 있어 자세 연습 구간을 다시 확인하세요.",
}


def reward_row(**overrides):
    """Create one editable Q-table row with unspecified actions set to 0."""
    row = {action: 0.0 for action in ACTIONS}
    row.update(overrides)
    return row


# Mock Q-table for Pose Agent development.
#
# Edit this dictionary directly when the Supervisor Agent's policy changes.
# Key: (pose_state, fail_count)
# Value: reward candidates for each supervisor action.
MOCK_POSE_Q_TABLE = {
    ("GOOD", 0): reward_row(
        **{
            "SA-17_POSITIVE_T": 1.0,
        }
    ),

    ("LEFT_WRIST", 0): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": 0.2,
        }
    ),
    ("LEFT_WRIST", 1): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": -0.3,
        }
    ),
    ("LEFT_WRIST", 2): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": -0.6,
        }
    ),
    ("LEFT_WRIST", 3): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": -1.0,
            "SA-18_SWITCH_POSTURE_REPEAT": 0.8,
        }
    ),

    ("RIGHT_WRIST", 0): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": 0.2,
        }
    ),
    ("RIGHT_WRIST", 1): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": -0.3,
        }
    ),
    ("RIGHT_WRIST", 2): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": -0.6,
        }
    ),
    ("RIGHT_WRIST", 3): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": -1.0,
            "SA-18_SWITCH_POSTURE_REPEAT": 0.8,
        }
    ),

    ("WRIST_SHAKE", 0): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": 0.2,
        }
    ),
    ("WRIST_SHAKE", 1): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": -0.3,
        }
    ),
    ("WRIST_SHAKE", 2): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": -0.6,
        }
    ),
    ("WRIST_SHAKE", 3): reward_row(
        **{
            "SA-12_WRIST_STRAIGHTEN": -1.0,
            "SA-18_SWITCH_POSTURE_REPEAT": 0.8,
        }
    ),

    ("ELBOW_WIDTH", 0): reward_row(
        **{
            "SA-13_ELBOW_RAISE": 0.2,
        }
    ),
    ("ELBOW_WIDTH", 1): reward_row(
        **{
            "SA-13_ELBOW_RAISE": -0.3,
        }
    ),
    ("ELBOW_WIDTH", 2): reward_row(
        **{
            "SA-13_ELBOW_RAISE": -0.6,
        }
    ),
    ("ELBOW_WIDTH", 3): reward_row(
        **{
            "SA-13_ELBOW_RAISE": -1.0,
            "SA-18_SWITCH_POSTURE_REPEAT": 0.8,
        }
    ),

    ("LEFT_ARM", 0): reward_row(
        **{
            "SA-13_ELBOW_RAISE": 0.2,
        }
    ),
    ("LEFT_ARM", 1): reward_row(
        **{
            "SA-13_ELBOW_RAISE": -0.3,
        }
    ),
    ("LEFT_ARM", 2): reward_row(
        **{
            "SA-13_ELBOW_RAISE": -0.6,
        }
    ),
    ("LEFT_ARM", 3): reward_row(
        **{
            "SA-13_ELBOW_RAISE": -1.0,
            "SA-18_SWITCH_POSTURE_REPEAT": 0.8,
        }
    ),

    ("RIGHT_ARM", 0): reward_row(
        **{
            "SA-13_ELBOW_RAISE": 0.2,
        }
    ),
    ("RIGHT_ARM", 1): reward_row(
        **{
            "SA-13_ELBOW_RAISE": -0.3,
        }
    ),
    ("RIGHT_ARM", 2): reward_row(
        **{
            "SA-13_ELBOW_RAISE": -0.6,
        }
    ),
    ("RIGHT_ARM", 3): reward_row(
        **{
            "SA-13_ELBOW_RAISE": -1.0,
            "SA-18_SWITCH_POSTURE_REPEAT": 0.8,
        }
    ),

    # 현재 Pose Agent feature에는 손가락/엄지/어깨 feature가 없지만,
    # Supervisor Agent의 action space에 맞춰 mock state를 미리 둔다.
    ("FINGER_CURVED", 0): reward_row(),
    ("FINGER_CURVED", 1): reward_row(
        **{
            "SA-14_FINGER_CURVE": -0.3,
        }
    ),
    ("FINGER_CURVED", 2): reward_row(
        **{
            "SA-14_FINGER_CURVE": -0.6,
        }
    ),
    ("FINGER_CURVED", 3): reward_row(
        **{
            "SA-14_FINGER_CURVE": -1.0,
            "SA-18_SWITCH_POSTURE_REPEAT": 0.8,
        }
    ),

    ("SHOULDER", 0): reward_row(
        **{
            "SA-15_SHOULDER_RELAX": 0.2,
        }
    ),
    ("SHOULDER", 1): reward_row(
        **{
            "SA-15_SHOULDER_RELAX": -0.3,
        }
    ),
    ("SHOULDER", 2): reward_row(
        **{
            "SA-15_SHOULDER_RELAX": -0.6,
        }
    ),
    ("SHOULDER", 3): reward_row(
        **{
            "SA-15_SHOULDER_RELAX": -1.0,
            "SA-18_SWITCH_POSTURE_REPEAT": 0.8,
        }
    ),

    ("THUMB_TENSION", 0): reward_row(
        **{
            "SA-16_THUMB_RELAX": 0.2,
        }
    ),
    ("THUMB_TENSION", 1): reward_row(
        **{
            "SA-16_THUMB_RELAX": -0.3,
        }
    ),
    ("THUMB_TENSION", 2): reward_row(
        **{
            "SA-16_THUMB_RELAX": -0.6,
        }
    ),
    ("THUMB_TENSION", 3): reward_row(
        **{
            "SA-16_THUMB_RELAX": -1.0,
            "SA-18_SWITCH_POSTURE_REPEAT": 0.8,
        }
    ),

    ("UNKNOWN", 0): reward_row(),
    ("UNKNOWN", 1): reward_row(),
    ("UNKNOWN", 2): reward_row(),
    ("UNKNOWN", 3): reward_row(
        **{
            "SA-18_SWITCH_POSTURE_REPEAT": 0.5,
        }
    ),
}


@dataclass(frozen=True)
class PoseQState:
    posture_state: str
    fail_count: int = 0

    def as_tuple(self):
        return (self.posture_state, self.fail_count)


class MockPoseRewardTable:
    """Thin wrapper around MOCK_POSE_Q_TABLE.

    The future Supervisor Agent will own the real Q-table. For now, the Pose
    Agent can use this class only to convert analysis results into table states
    and look up reward values.
    """

    def __init__(self, q_table=None, max_fail_count=3, default_reward=0.0):
        self.q_table = q_table or MOCK_POSE_Q_TABLE
        self.max_fail_count = max_fail_count
        self.default_reward = default_reward

    def get_state(self, analysis_result, previous_state=None):
        previous_state = self._to_pose_q_state(previous_state)
        posture_case = analysis_result.get("case")

        if posture_case == "안정":
            return PoseQState("GOOD", 0)

        posture_state = self._get_primary_issue_state(analysis_result)
        fail_count = 1

        if previous_state and previous_state.posture_state == posture_state:
            fail_count = min(previous_state.fail_count + 1, self.max_fail_count)

        return PoseQState(posture_state, fail_count)

    def get_reward(self, state, action):
        state_key = self._normalize_state(state)
        row = self.q_table.get(state_key)

        if row is None:
            return self.default_reward

        return row.get(action, self.default_reward)

    def get_action_rewards(self, state):
        state_key = self._normalize_state(state)
        return dict(self.q_table.get(state_key, reward_row()))

    def best_action(self, state):
        rewards = self.get_action_rewards(state)
        return max(rewards, key=rewards.get)

    def decide_action(self, state):
        state_key = self._normalize_state(state)
        posture_state = state_key[0]
        return STATE_TO_ACTION.get(posture_state, self.best_action(state_key))

    def get_feedback(self, action):
        return ACTION_FEEDBACK.get(action, "")

    def lookup_from_analysis(self, analysis_result, action, previous_state=None):
        state = self.get_state(analysis_result, previous_state)
        reward = self.get_reward(state, action)

        return {
            "state": state.as_tuple(),
            "action": action,
            "reward": reward,
        }

    def transition_lookup(self, previous_result, current_result, action, previous_state=None):
        if previous_state is None:
            previous_state = self.get_state(previous_result)

        current_state = self.get_state(current_result, previous_state)
        reward = self.get_transition_reward(previous_state, current_state, action)

        return {
            "previous_state": previous_state.as_tuple(),
            "current_state": current_state.as_tuple(),
            "action": action,
            "reward": reward,
        }

    def get_transition_reward(self, previous_state, current_state, action):
        previous_state = self._to_pose_q_state(previous_state)
        current_state = self._to_pose_q_state(current_state)

        if (
            previous_state.posture_state != "GOOD"
            and current_state.posture_state == "GOOD"
        ):
            return 1.0

        return self.get_reward(current_state, action)

    def build_transition_report(self, current_result, previous_result=None, previous_state=None):
        """Build the Pose Agent payload that can be sent to Supervisor Agent.

        current_result and previous_result are outputs from PostureFeedbackAnalyzer.
        The returned dict contains state measurement, fail count, action judgment,
        user-facing feedback, and reward lookup.
        """
        previous_state = self._to_pose_q_state(previous_state)

        if previous_state is None and previous_result is not None:
            previous_state = self.get_state(previous_result)

        current_state = self.get_state(current_result, previous_state)
        action = self.decide_action(current_state)
        feedback = self._extract_feedback(current_result, action)

        if previous_state is None:
            reward = self.get_reward(current_state, action)
            previous_state_tuple = None
        else:
            reward = self.get_transition_reward(previous_state, current_state, action)
            previous_state_tuple = previous_state.as_tuple()

        return {
            "state_measurement": current_state.posture_state,
            "state": current_state.as_tuple(),
            "previous_state": previous_state_tuple,
            "fail_count": current_state.fail_count,
            "action": action,
            "feedback": feedback,
            "user_message": feedback,
            "reward": reward,
            "posture_score": current_result.get("final_score"),
            "posture_case": current_result.get("case"),
            "send_to_supervisor": {
                "pose_state": current_state.posture_state,
                "fail_count": current_state.fail_count,
                "action": action,
                "reward": reward,
                "posture_score": current_result.get("final_score"),
                "posture_case": current_result.get("case"),
            },
        }

    @staticmethod
    def _get_primary_issue_state(analysis_result):
        explicit_state = (
            analysis_result.get("pose_state")
            or analysis_result.get("posture_state")
            or analysis_result.get("state")
        )
        if explicit_state:
            return explicit_state

        details = analysis_result.get("details", [])
        primary_issue = details[0] if details else None
        feature_index = primary_issue.get("feature_index") if primary_issue else None
        return FEATURE_TO_STATE.get(feature_index, "UNKNOWN")

    def _extract_feedback(self, analysis_result, action):
        summary = analysis_result.get("summary", {})
        coaching = summary.get("coaching", [])

        if coaching:
            return coaching[0]

        return self.get_feedback(action)

    @staticmethod
    def _normalize_state(state):
        if isinstance(state, PoseQState):
            return state.as_tuple()

        if isinstance(state, tuple):
            return state

        raise TypeError("state must be PoseQState or (posture_state, fail_count) tuple")

    @staticmethod
    def _to_pose_q_state(state):
        if state is None:
            return None

        if isinstance(state, PoseQState):
            return state

        if isinstance(state, tuple):
            return PoseQState(state[0], state[1])

        raise TypeError("state must be PoseQState, tuple, or None")
