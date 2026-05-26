from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PostureFeedbackAnalyzer:
    """Analyze pose feature sequences and return posture feedback.

    Input feature order:
    0. wrist_to_shoulder_left
    1. elbow_width
    2. wrist_velocity_left
    3. left_elbow_angle
    4. right_elbow_angle
    5. right_wrist_angle
    """

    FEATURE_CONTEXT = {
        0: "왼손목-왼어깨 거리",
        1: "양쪽 팔꿈치 거리",
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
        4: "high",
        5: "both",
    }

    FEATURE_FEEDBACK = {
        0: "왼손과 어깨 간 정렬이 무너지고 있습니다.",
        1: "양 팔의 간격 균형이 깨지고 있습니다.",
        2: "손목 움직임이 과도하게 흔들리고 있습니다.",
        3: "왼팔 각도가 비정상적으로 변형되고 있습니다.",
        4: "오른팔이 과도하게 벌어지고 있습니다.",
        5: "오른손목 정렬이 무너지고 있습니다.",
    }

    COACHING_RULES = {
        0: lambda z: "왼손 위치를 몸 중심 방향으로 안정화하세요.",
        1: lambda z: "양 팔 간격을 일정하게 유지하세요.",
        2: lambda z: "손목에 힘이 들어가 흔들림이 커지고 있습니다.",
        3: lambda z: "왼팔 각도를 일정하게 유지하세요.",
        4: lambda z: "오른팔이 과도하게 벌어지고 있습니다.",
        5: lambda z: "오른손목 정렬이 틀어지고 있습니다.",
    }

    def __init__(self, data_dir=None, good_count=4, bad_file="dataset_bad_1.npz"):
        self.data_dir = self._resolve_data_dir(data_dir)
        self.good_count = good_count
        self.bad_file = bad_file

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
            )),
        ])

        self.feature_mean = None
        self.feature_std = None
        self.fit()

    @staticmethod
    def _resolve_data_dir(data_dir):
        if data_dir is not None:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            return data_dir

        candidates = [
            Path("data/datasets"),
            Path("../datasets"),
            Path("../../data/datasets"),
            Path("/content"),
        ]

        for candidate in candidates:
            if (candidate / "dataset_good_1.npz").exists():
                return candidate

        raise FileNotFoundError(
            "dataset_good_1.npz를 찾을 수 없습니다. data_dir를 직접 지정하세요."
        )

    @staticmethod
    def summarize_sequence(sequence):
        """Convert a 30-frame feature sequence into one 6-feature vector."""
        sequence = np.asarray(sequence, dtype=np.float32)
        return np.median(sequence, axis=-2)

    def load_dataset(self):
        good_sequences = []

        for i in range(1, self.good_count + 1):
            path = self.data_dir / f"dataset_good_{i}.npz"
            data = np.load(path)
            good_sequences.append(data["sequences"])

        bad_path = self.data_dir / self.bad_file
        bad_data = np.load(bad_path)

        all_good = np.concatenate(good_sequences, axis=0)
        all_bad = bad_data["sequences"]

        good_summary = self.summarize_sequence(all_good)
        bad_summary = self.summarize_sequence(all_bad)

        x = np.vstack([good_summary, bad_summary])
        y = np.concatenate([
            np.zeros(len(good_summary), dtype=int),
            np.ones(len(bad_summary), dtype=int),
        ])

        return x, y

    def fit(self):
        x, y = self.load_dataset()
        self.model.fit(x, y)

        good_samples = x[y == 0]
        self.feature_mean = np.mean(good_samples, axis=0)
        self.feature_std = np.std(good_samples, axis=0)

        return self

    def analyze(self, sample):
        """Return posture score, case, and feature-level feedback.

        sample can be either:
        - shape (6,): already summarized feature vector
        - shape (30, 6): feature sequence from FeatureExtractor.update_buffer()
        """
        feature_vector = self._to_feature_vector(sample)
        bad_prob = self._predict_bad_probability(feature_vector)

        explanations = []
        biomechanical_risk = 0.0
        severe_count = 0
        danger_count = 0

        scaled_sample = self.model.named_steps["scaler"].transform(
            feature_vector.reshape(1, -1)
        )[0]
        weights = self.model.named_steps["classifier"].coef_[0]

        for idx, value in enumerate(feature_vector):
            z_score = (value - self.feature_mean[idx]) / (self.feature_std[idx] + 1e-8)
            risk_z = self._compute_directional_risk(idx, z_score)
            status = self._classify_deviation(risk_z)

            contribution = scaled_sample[idx] * weights[idx]
            bad_contribution = max(0.0, float(contribution))

            biomechanical_risk += risk_z

            if status == "심각":
                severe_count += 1
            elif status == "위험":
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
                "warning": self.FEATURE_FEEDBACK[idx],
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
                item["bad_contribution"] / (total_bad_contribution + 1e-8) * 100
            )

        explanations = sorted(
            explanations,
            key=lambda item: (item["risk_z"], item["bad_contribution"]),
            reverse=True,
        )

        final_score = self._compute_hybrid_score(bad_prob, biomechanical_risk)
        posture_case = self._determine_posture_case(
            final_score,
            severe_count,
            danger_count,
        )

        return {
            "case": posture_case,
            "final_score": float(round(final_score, 2)),
            "classifier_probability": float(round(bad_prob * 100, 2)),
            "biomechanical_risk": float(round(biomechanical_risk, 4)),
            "details": explanations,
            "summary": self._build_summary(posture_case, explanations),
        }

    def analyze_many(self, samples):
        samples = np.asarray(samples, dtype=np.float32)

        if samples.ndim not in (2, 3):
            raise ValueError("samples must have shape (N, 6) or (N, 30, 6)")

        return [self.analyze(sample) for sample in samples]

    def _to_feature_vector(self, sample):
        sample = np.asarray(sample, dtype=np.float32)

        if sample.shape == (6,):
            return sample

        if sample.ndim == 2 and sample.shape[1] == 6:
            return self.summarize_sequence(sample)

        raise ValueError(
            "sample must have shape (6,) or (sequence_length, 6). "
            f"Got {sample.shape}."
        )

    def _predict_bad_probability(self, feature_vector):
        return float(self.model.predict_proba(feature_vector.reshape(1, -1))[0][1])

    def _compute_directional_risk(self, idx, z_score):
        direction = self.FEATURE_DIRECTIONS[idx]

        if direction == "high":
            return max(0.0, float(z_score))

        if direction == "low":
            return max(0.0, float(-z_score))

        return abs(float(z_score))

    @staticmethod
    def _classify_deviation(risk_z):
        if risk_z < 1:
            return "정상"

        if risk_z < 2:
            return "주의"

        if risk_z < 3:
            return "위험"

        return "심각"

    @staticmethod
    def _compute_hybrid_score(classifier_probability, biomechanical_risk):
        global_risk = classifier_probability * 100
        biomech_score = min(100, biomechanical_risk * 20)
        return 0.6 * global_risk + 0.4 * biomech_score

    @staticmethod
    def _determine_posture_case(final_score, severe_count, danger_count):
        if severe_count >= 1:
            return "위험"

        if danger_count >= 2:
            return "위험"

        if final_score < 30:
            return "안정"

        if final_score < 60:
            return "불안정"

        return "위험"

    @staticmethod
    def _build_summary(posture_case, explanations, top_k=3):
        coaching = [
            item["coaching"]
            for item in explanations[:top_k]
            if item["coaching"] is not None
        ]

        if not coaching:
            coaching = ["현재 자세는 안정적으로 유지되고 있습니다."]

        return {
            "posture": posture_case,
            "top_issues": [
                {
                    "feature": item["feature"],
                    "status": item["status"],
                    "risk_percent": float(round(item["risk_percent"], 2)),
                    "coaching": item["coaching"],
                }
                for item in explanations[:top_k]
            ],
            "coaching": coaching,
        }
