from pathlib import Path

import joblib
import numpy as np


class PoseFeedbackAnalyzer:
    """
    Runtime posture analyzer.

    Input:
        (6,)      -> summarized feature vector
        (30, 6)   -> sequence feature input

    Output:
        posture feedback + risk analysis
    """

    FEATURE_CONTEXT = {
        0: "왼손목-왼어깨 거리",
        1: "양쪽 어깨의 높이 차이",
        2: "왼손목 속도",
        3: "왼팔 각도",
        4: "오른팔 각도",
        5: "오른손목 각도",
    }

    FEATURE_DIRECTIONS = {
        0: "both",
        1: "both",
        2: "high",
        3: "both",
        4: "low",
        5: "both",
    }

    FEATURE_PROBLEMS = {
        0: "왼손 포지션이 흔들리며 손과 어깨의 정렬이 불안정합니다.",
        1: "양쪽 어깨 높이 차이가 발생하여 상체 균형이 무너지고 있습니다.",
        2: "왼손목 움직임이 과도하여 운지 안정성이 저하되고 있습니다.",
        3: "왼팔 자세가 흐트러지며 바이올린 지지 자세가 불안정합니다.",
        4: "오른팔 각도가 감소하며 보잉 자세가 무너지고 있습니다.",
        5: "오른손목 정렬이 흐트러져 활의 움직임이 불안정합니다.",
    }

    COACHING_RULES = {
        0: lambda z: "왼어깨에 힘을 빼고 손과 어깨의 거리를 일정하게 유지하세요.",
        1: lambda z: "양쪽 어깨의 높이를 일정하게 유지하세요.",
        2: lambda z: "왼손목의 힘을 빼고 손가락 중심으로 운지하세요.",
        3: lambda z: "왼팔의 각도를 일정하게 유지하여 악기를 안정적으로 지지하세요.",
        4: lambda z: "오른팔이 접히고 있습니다. 팔꿈치 높이를 유지하며 자연스럽게 보잉하세요.",
        5: lambda z: "오른손목에 힘을 빼고 활이 직선으로 움직일 수 있도록 정렬을 유지하세요.",
    }

    def __init__(self, model_path="model/pose_model.pkl"):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.feature_mean = bundle["feature_mean"]
        self.feature_std = bundle["feature_std"]

    @staticmethod
    def summarize_sequence(sequence):
        """
        Convert sequence (30, 6) -> feature vector (6,)
        """

        sequence = np.asarray(sequence, dtype=np.float32)

        return np.median(sequence, axis=-2)

    def analyze(self, sample):
        """
        Analyze posture sample.

        Input:
            sample.shape == (6,)
            or
            sample.shape == (30, 6)
        """

        feature_vector = self._to_feature_vector(sample)

        bad_probability = self._predict_bad_probability(
            feature_vector
        )

        explanations = []

        biomechanical_risk = 0.0

        severe_count = 0
        danger_count = 0

        scaled_sample = (
            self.model.named_steps["scaler"]
            .transform(feature_vector.reshape(1, -1))[0]
        )

        weights = (
            self.model.named_steps["classifier"]
            .coef_[0]
        )

        for idx, value in enumerate(feature_vector):

            z_score = (
                (value - self.feature_mean[idx])
                / (self.feature_std[idx] + 1e-8)
            )

            risk_z = self._compute_directional_risk(
                idx,
                z_score,
            )

            status = self._classify_deviation(risk_z)

            contribution = (
                scaled_sample[idx] * weights[idx]
            )

            bad_contribution = max(
                0.0,
                float(contribution),
            )

            biomechanical_risk += risk_z / len(self.FEATURE_CONTEXT)

            if status == "심각":
                severe_count += 1

            elif status == "주의":
                danger_count += 1

            explanations.append({
                "feature_index": idx,
                "feature": self.FEATURE_CONTEXT[idx],
                "value": float(value),
                "z_score": float(z_score),
                "risk_z": float(risk_z),
                "status": status,
                "contribution": float(contribution),
                "bad_contribution": bad_contribution,
                "warning": self.FEATURE_PROBLEMS[idx],
                "coaching": (
                    self.COACHING_RULES[idx](z_score)
                    if risk_z >= 2
                    else None
                ),
            })

        total_bad_contribution = sum(
            item["bad_contribution"]
            for item in explanations
        )

        for item in explanations:
            item["risk_percent"] = float(
                item["bad_contribution"]
                / (total_bad_contribution + 1e-8)
                * 100
            )

        explanations = sorted(
            explanations,
            key=lambda item: (
                item["risk_z"],
                item["bad_contribution"],
            ),
            reverse=True,
        )

        final_score = self._compute_hybrid_score(
            bad_probability,
            biomechanical_risk,
        )

        posture_case = self._determine_posture_case(
            final_score,
            severe_count,
            danger_count,
        )

        return {
            "case": posture_case,
            "final_score": float(round(final_score, 2)),
            "classifier_probability": float(
                round(bad_probability * 100, 2)
            ),
            "biomechanical_risk": float(
                round(biomechanical_risk, 4)
            ),
            "details": explanations,
            "summary": self._build_summary(
                posture_case,
                explanations,
            ),
        }

    def analyze_many(self, samples):
        """
        Batch analysis.
        """

        samples = np.asarray(samples, dtype=np.float32)

        if samples.ndim not in (2, 3):
            raise ValueError(
                "samples must have shape "
                "(N, 6) or (N, 30, 6)"
            )

        return [
            self.analyze(sample)
            for sample in samples
        ]

    def _to_feature_vector(self, sample):
        sample = np.asarray(sample, dtype=np.float32)

        if sample.shape == (6,):
            return sample

        if sample.ndim == 2 and sample.shape[1] == 6:
            return self.summarize_sequence(sample)

        raise ValueError(
            "sample must have shape "
            "(6,) or (sequence_length, 6). "
            f"Got {sample.shape}."
        )

    def _predict_bad_probability(self, feature_vector):

        return float(
            self.model.predict_proba(
                feature_vector.reshape(1, -1)
            )[0][1]
        )

    def _compute_directional_risk(
        self,
        idx,
        z_score,
    ):
        direction = self.FEATURE_DIRECTIONS[idx]

        if direction == "high":
            return max(0.0, float(z_score))

        if direction == "low":
            return max(0.0, float(-z_score))

        return abs(float(z_score))

    @staticmethod
    def _classify_deviation(risk_z):

        if risk_z <= 1.5:
            return "정상"

        if risk_z <= 3:
            return "주의"

        return "심각"

    @staticmethod
    def _compute_hybrid_score(
        classifier_probability,
        biomechanical_risk,
    ):
        global_risk = classifier_probability * 100

        biomech_score = min(
            100,
            biomechanical_risk / len(PoseFeedbackAnalyzer.FEATURE_CONTEXT) * 100,
        )

        return (
            0.6 * global_risk
            + 0.4 * biomech_score
        )

    @staticmethod
    def _determine_posture_case(
        final_score,
        severe_count,
        danger_count,
    ):

        if severe_count >= 3:
            return "주의"

        if danger_count >= 2:
            return "주의"

        if final_score < 45:
            return "안정"

        if final_score < 60:
            return "주의"

        return "위험"

    @staticmethod
    def _build_summary(
        posture_case,
        explanations,
        top_k=3,
    ):

        coaching = [
            item["coaching"]
            for item in explanations[:top_k]
            if item["coaching"] is not None
        ]

        if not coaching:
            coaching = [
                "현재 자세는 안정적으로 유지되고 있습니다."
            ]

        return {
            "posture": posture_case,
            "top_issues": [
                {
                    "feature": item["feature"],
                    "status": item["status"],
                    "risk_percent": float(
                        round(item["risk_percent"], 2)
                    ),
                    "coaching": item["coaching"],
                }
                for item in explanations[:top_k]
            ],
            "coaching": coaching,
        }


if __name__ == "__main__":

    analyzer = PoseFeedbackAnalyzer(
        "model/pose_model.pkl"
    )

    dummy_sequence = np.random.rand(30, 6)

    result = analyzer.analyze(dummy_sequence)

    from pprint import pprint

    pprint(result)

