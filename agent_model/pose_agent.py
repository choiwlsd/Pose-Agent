import json
import time

import cv2

from agent_model.feature_extractor import FeatureExtractor
from agent_model.pose_extractor import PoseExtractor
from agent_model.pose_analyzer import PoseFeedbackAnalyzer
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
        self.supervisor = supervisor

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
        feedback = result["summary"]
        state, reward_info = self.update_supervisor(result)

        self.prev_result = result
        self.prev_state = state

        self.print_feedback(result, reward_info)
        self.write_result(result, feedback, reward_info)

        return {
            "features": features,
            "analysis": result,
            "feedback": feedback,
            "state": state.as_tuple() if state else None,
            "reward": reward_info,
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

    def print_feedback(self, result, reward_info=None):
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

    def write_result(self, result, feedback, reward_info):
        if not self.output_path:
            return

        output = {
            "timestamp": time.time(),
            "case": result["case"],
            "final_score": result["final_score"],
            "biomechanical_risk": result["biomechanical_risk"],
            "top_issues": feedback["top_issues"],
            "coaching": feedback["coaching"],
            "supervisor": reward_info,
        }

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False, indent=4) + "\n")

    def release(self):
        self.pose_extractor.release()
        self.webcam.release()
        cv2.destroyAllWindows()
