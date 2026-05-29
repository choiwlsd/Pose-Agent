import json
import time

import cv2

from agent_model.feature_extractor import FeatureExtractor
from agent_model.pose_extractor import PoseExtractor
from agent_model.pose_analyzer import PoseFeedbackAnalyzer
from agent_model.supervisor_reward import MockPoseRewardTable
from tools.visualizer import print_features


class PoseAgent:
    def __init__(
        self,
        webcam,
        pose_extractor=None,
        feature_extractor=None,
        analyzer=None,
        output_path="pose_output.json",
        supervisor=None,
    ):
        self.webcam = webcam
        self.pose_extractor = pose_extractor or PoseExtractor(webcam)
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.analyzer = analyzer or PoseFeedbackAnalyzer()
        self.output_path = output_path
        self.supervisor = supervisor or MockPoseRewardTable()

        self.prev_result = None
        self.prev_state = None

    def run(self, display=True):
        try:
            while self.webcam.isOpened():
                ret, frame = self.webcam.read()
                if not ret:
                    break

                landmarks, frame = self.pose_extractor.extract(frame)
                self.step(landmarks)

                if display:
                    cv2.imshow("PoseAgent", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.release()

    def step(self, landmarks, print_debug=True):
        if landmarks is None:
            return None

        features = self.pose_to_features(landmarks, print_debug=print_debug)
        sequence = self.feature_extractor.update_buffer(features)

        if sequence is None:
            return {
                "features": features,
                "analysis": None,
                "feedback": None,
                "state": None,
                "reward": None,
            }

        result = self.analyzer.analyze(sequence)

        if self.prev_state is not None:
            transition = self.supervisor.evaluate_transition(
                previous_result=self.prev_result,
                current_result=result,
                previous_state=self.prev_state, 
            )
        else:
            transition = None

        feedback = result["summary"]

        # supervisor state 변환
        current_state = self.supervisor.get_state(result, self.prev_state)

        # reward + action 계산
        if self.prev_result is not None:
            reward_info = self.supervisor.compute_reward(
                self.prev_result,
                result,
                self.prev_state,
                current_state,
            )

            action = self.supervisor.choose_action(self.prev_state)

            self.supervisor.update(
                self.prev_state,
                action,
                reward_info["reward"],
                current_state,
            )
        else:
            reward_info = None
            action = None

        self.prev_result = result
        self.prev_state = current_state

        self.print_feedback(result, transition)
        self.write_result(result, feedback, transition)

        return {
            "features": features,
            "analysis": result,
            "feedback": feedback,
            "state": current_state.as_tuple() if current_state else None,
            "action": action,
            "transition": transition,
        }

    def pose_to_features(self, landmarks, print_debug=True):
        features = self.feature_extractor.compute(landmarks)
        if print_debug:
            print_features(features)
        return features

    def update_supervisor(self, result):
        if not self.supervisor:
            return None, None

        state = self.supervisor.get_state(result, previous_state=self.prev_state)
        if self.prev_result is None:
            return state, None

        reward_info = self.supervisor.compute_reward(
            self.prev_result,
            result,
            self.prev_state,
            state,
        )
        return state, reward_info

    def print_feedback(self, result, reward_info=None, transition=None):
        print("\n" + "=" * 60)
        print(f"[POSTURE CASE] {result['case']}")
        print(f"[FINAL SCORE ] {result['final_score']}")
        print(f"[RISK         ] {result['biomechanical_risk']}")

        print("\n[TOP ISSUES]")
        for item in result["summary"]["top_issues"]:
            print(f"- {item['feature']} | {item['status']} | {item['risk_percent']}%")

        print("\n[COACHING]")
        for coaching in result["summary"]["coaching"]:
            print(f"- {coaching}")

        if reward_info:
            print("\n[SUPERVISOR REWARD]")
            print(f"- reward: {reward_info['reward']}")
            print(f"- reason: {reward_info['reason']}")

        if transition:
            print("\n[RL TRANSITION]")
            print(f"- previous_state: {transition['previous_state']}")
            print(f"- current_state: {transition['current_state']}")
            print(f"- action: {transition['action']}")
            print(f"- reward: {transition['reward']}")
            print(f"- reason: {transition['reason']}")
            print(f"- score_delta: {transition['score_delta']}")
            print(f"- updated_q: {transition['updated_q']}")

    def write_result(self, result, feedback, transition=None):
        if not self.output_path:
            return

        output = {
            "timestamp": time.time(),
            "case": result["case"],
            "final_score": result["final_score"],
            "biomechanical_risk": result["biomechanical_risk"],
            "top_issues": feedback["top_issues"],
            "coaching": feedback["coaching"],
            "transition": transition,
        }

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False, indent=4) + "\n")

    def release(self):
        self.pose_extractor.release()
        self.webcam.release()
        cv2.destroyAllWindows()
