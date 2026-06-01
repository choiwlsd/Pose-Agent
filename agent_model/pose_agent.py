import json

import cv2

from agent_model.feature_extractor import FeatureExtractor
from agent_model.pose_extractor import PoseExtractor
from agent_model.pose_analyzer import PoseFeedbackAnalyzer
from agent_model.supervisor_reward import MockPoseRewardTable
from tools.visualizer import print_features


class PoseAgent:
    def __init__(
        self,
        source,
        pose_extractor=None,
        feature_extractor=None,
        analyzer=None,
        output_path="pose_output.json",
        supervisor=None,
        feedback_interval_seconds=5.0,
    ):
        self.source = source
        self.fps = self._get_capture_fps()

        # 몇 초마다 피드백을 줄지 설정 (현재: 5초마다)
        self.feedback_interval_seconds = feedback_interval_seconds
        # 다음 피드백 타임스탬프 초기화
        self.next_feedback_timestamp = feedback_interval_seconds
        self.frame_index = 0

        # 컴포넌트 초기화
        self.pose_extractor = pose_extractor or PoseExtractor(self.source)
        self.feature_extractor = feature_extractor or FeatureExtractor(
            sequence_length=max(1, int(round(self.fps * feedback_interval_seconds))),
            stride=1,
        )
        self.analyzer = analyzer or PoseFeedbackAnalyzer()
        self.output_path = output_path
        self.supervisor = supervisor or MockPoseRewardTable()

        self.prev_result = None
        self.prev_state = None

    # 메인 루프: 웹캠 프레임 읽기 > 자세 분석(step) > 피드백 출력
    def run(self, display=True):
        try:
            while self.source.isOpened():
                ret, frame = self.source.read()
                if not ret:
                    break

                self.frame_index += 1
                landmarks, frame = self.pose_extractor.extract(frame)
                self.step(landmarks)

                if display:
                    cv2.imshow("PoseAgent", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.release()

    # 한 프레임의 landmakrs 입력 > 자세 분석 + 피드백
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

        timestamp = self.current_video_timestamp()
        if timestamp < self.next_feedback_timestamp:
            return {
                "features": features,
                "analysis": None,
                "feedback": None,
                "state": None,
                "reward": None,
            }

        # 자세 분석
        result = self.analyzer.analyze(sequence)

        # 이전 상태가 존재하면 상태 전이 평가
        if self.prev_state is not None:
            transition = self.supervisor.evaluate_transition(
                previous_result=self.prev_result,
                current_result=result,
                previous_state=self.prev_state, 
            )
        else:
            transition = None

        # 사용자에게 보여줄 요약 피드백
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

        # 다음 피드백 타임스탬프 업데이트
        self.prev_result = result
        self.prev_state = current_state
        while self.next_feedback_timestamp <= timestamp:
            self.next_feedback_timestamp += self.feedback_interval_seconds

        # console 출력 + 결과 저장
        self.print_feedback(result, reward_info=reward_info, transition=transition)
        self.write_result(result, feedback, transition, timestamp)

        return {
            "features": features,
            "analysis": result,
            "feedback": feedback,
            "state": current_state.as_tuple() if current_state else None,
            "action": action,
            "transition": transition,
        }

    # 웹캠 FPS 가져오기 (기본값: 30 FPS)
    def _get_capture_fps(self):
        fps = self.source.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            return fps
        return 30.0

    def current_video_timestamp(self):
        timestamp = self.source.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if timestamp > 0:
            return timestamp
        return self.frame_index / self.fps

    def pose_to_features(self, landmarks, print_debug=True):
        features = self.feature_extractor.compute(landmarks)
        if print_debug:
            print_features(features)
        return features

    # 현재 결과 활용 > 상태 및 Reward 계산
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

    # 결과 출력
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

    # 결과를 JSON 파일로 저장
    def write_result(self, result, feedback, transition=None, timestamp=None):
        if not self.output_path:
            return

        if timestamp is None:
            timestamp = self.current_video_timestamp()

        output = {
            "timestamp": timestamp,
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
        self.source.release()
        cv2.destroyAllWindows()
