from dataclasses import dataclass

import numpy as np


DEFAULT_ACTIONS = [
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


STATE_TO_RECOMMENDED_ACTION = {
    "LEFT_WRIST": "SA-12_WRIST_STRAIGHTEN",
    "RIGHT_WRIST": "SA-12_WRIST_STRAIGHTEN",
    "WRIST_SHAKE": "SA-12_WRIST_STRAIGHTEN",
    "ELBOW_WIDTH": "SA-13_ELBOW_RAISE",
    "LEFT_ARM": "SA-13_ELBOW_RAISE",
    "RIGHT_ARM": "SA-13_ELBOW_RAISE",
    "SHOULDER": "SA-15_SHOULDER_RELAX",
    "GOOD": "SA-17_POSITIVE_T",
}


@dataclass(frozen=True)
class PoseQState:
    posture_state: str
    fail_count: int

    def as_tuple(self):
        return (self.posture_state, self.fail_count)

# Q-table 예시 
class MockPoseRewardTable:
    """Mock Q-table and reward helper for the future Supervisor Agent.

    This class is intentionally small and deterministic. It does not replace a
    real Supervisor Agent; it gives the Pose Agent side a stable contract while
    the Q-learning loop is still being designed.
    """

    def __init__(
        self,
        actions=None,
        max_fail_count=3,
        alpha=0.3,
        gamma=0.9,
        epsilon=0.1,
    ):
        self.actions = list(actions or DEFAULT_ACTIONS)
        self.max_fail_count = max_fail_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = self._build_mock_q_table()

    def get_state(self, analysis_result, previous_state=None):
        """Convert PostureFeedbackAnalyzer output into a Q-table state."""
        posture_case = analysis_result.get("case")

        if posture_case == "안정":
            return PoseQState("GOOD", 0)

        details = analysis_result.get("details", [])
        primary_issue = details[0] if details else None
        feature_index = primary_issue.get("feature_index") if primary_issue else None
        posture_state = FEATURE_TO_STATE.get(feature_index, "UNKNOWN")

        fail_count = 0
        if previous_state and previous_state.posture_state == posture_state:
            fail_count = min(previous_state.fail_count + 1, self.max_fail_count)

        return PoseQState(posture_state, fail_count)

    def compute_reward(self, previous_result, current_result, previous_state, current_state):
        """Reward is positive when posture improves and negative when it worsens."""
        previous_score = float(previous_result.get("final_score", 100.0))
        current_score = float(current_result.get("final_score", 100.0))
        score_delta = previous_score - current_score

        if current_state.posture_state == "GOOD":
            reward = 1.0
            reason = "posture became stable"
        elif score_delta >= 10:
            reward = 0.6
            reason = "posture score improved clearly"
        elif score_delta >= 3:
            reward = 0.3
            reason = "posture score improved slightly"
        elif score_delta <= -10:
            reward = -0.7
            reason = "posture score became clearly worse"
        elif score_delta <= -3:
            reward = -0.4
            reason = "posture score became slightly worse"
        else:
            reward = -0.1
            reason = "posture did not meaningfully change"

        if (
            current_state.posture_state != "GOOD"
            and previous_state.posture_state == current_state.posture_state
            and current_state.fail_count >= self.max_fail_count
        ):
            reward -= 0.3
            reason += "; same issue repeated too many times"

        return {
            "reward": round(reward, 3),
            "reason": reason,
            "score_delta": round(score_delta, 3),
            "previous_state": previous_state.as_tuple(),
            "current_state": current_state.as_tuple(),
        }

    def choose_action(self, state, explore=False):
        """Pick an action from the mock Q-table."""
        self._ensure_state(state)

        if explore and np.random.random() < self.epsilon:
            return str(np.random.choice(self.actions))

        values = self.q_table[state.as_tuple()]
        best_index = int(np.argmax([values[action] for action in self.actions]))
        return self.actions[best_index]

    def update(self, state, action, reward, next_state):
        """Apply a simple Q-learning update to the mock table."""
        self._ensure_state(state)
        self._ensure_state(next_state)
        self._ensure_action(action)

        state_key = state.as_tuple()
        next_key = next_state.as_tuple()

        old_value = self.q_table[state_key][action]
        next_best = max(self.q_table[next_key].values())

        new_value = old_value + self.alpha * (
            reward + self.gamma * next_best - old_value
        )
        self.q_table[state_key][action] = round(new_value, 4)

        return self.q_table[state_key][action]

    def evaluate_transition(self, previous_result, current_result, previous_state=None, action=None):
        """Convenience method: state conversion, reward, action choice, and update."""
        if previous_state is None:
            previous_state = self.get_state(previous_result)

        current_state = self.get_state(current_result, previous_state)

        if action is None:
            action = self.choose_action(previous_state)

        reward_info = self.compute_reward(
            previous_result,
            current_result,
            previous_state,
            current_state,
        )
        updated_q = self.update(
            previous_state,
            action,
            reward_info["reward"],
            current_state,
        )

        return {
            "previous_state": previous_state.as_tuple(),
            "current_state": current_state.as_tuple(),
            "action": action,
            "reward": reward_info["reward"],
            "reason": reward_info["reason"],
            "score_delta": reward_info["score_delta"],
            "updated_q": updated_q,
        }

    def _build_mock_q_table(self):
        posture_states = sorted(set(FEATURE_TO_STATE.values()) | {"GOOD", "UNKNOWN", "SHOULDER"})
        table = {}

        for posture_state in posture_states:
            fail_range = [0] if posture_state == "GOOD" else range(self.max_fail_count + 1)

            for fail_count in fail_range:
                state_key = (posture_state, fail_count)
                table[state_key] = {action: 0.0 for action in self.actions}

                recommended_action = STATE_TO_RECOMMENDED_ACTION.get(posture_state)
                if recommended_action in table[state_key]:
                    table[state_key][recommended_action] = round(0.2 * fail_count, 2)

                if posture_state == "GOOD":
                    table[state_key]["SA-17_POSITIVE_T"] = 0.8

                if fail_count >= self.max_fail_count:
                    table[state_key]["SA-18_SWITCH_POSTURE_REPEAT"] = 0.8

        return table

    def _ensure_state(self, state):
        state_key = state.as_tuple()
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.actions}

    def _ensure_action(self, action):
        if action not in self.actions:
            raise ValueError(f"Unknown action: {action}")
