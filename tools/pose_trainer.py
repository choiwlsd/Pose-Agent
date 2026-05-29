from pathlib import Path
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PoseTrainer:
    """
    Train posture classification model and export runtime bundle.

    Dataset format:
        dataset_good_1.npz
        dataset_good_2.npz
        ...
        dataset_bad_1.npz

    Each .npz file must contain:
        sequences: (N, 30, 6)
    """

    def __init__(
        self,
        data_dir=None,
        good_count=4,
        bad_file="dataset_bad_1.npz",
    ):
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

    @staticmethod
    def _resolve_data_dir(data_dir):
        if data_dir is not None:
            data_dir = Path(data_dir)

            if not data_dir.exists():
                raise FileNotFoundError(
                    f"Data directory not found: {data_dir}"
                )

            return data_dir

        candidates = [
            Path("data/datasets"),
            Path("../datasets"),
            Path("../../data/datasets"),
            Path("/content"), # Colab 환경에서의 기본 경로
        ]

        for candidate in candidates:
            if (candidate / "dataset_good_1.npz").exists():
                return candidate

        raise FileNotFoundError(
            "dataset_good_1.npz를 찾을 수 없습니다."
        )

    @staticmethod
    def summarize_sequence(sequence):
        """
        Convert sequence (30, 6) -> feature vector (6,)
        """
        sequence = np.asarray(sequence, dtype=np.float32)

        return np.median(sequence, axis=-2)

    def load_dataset(self):
        good_sequences = []

        for i in range(1, self.good_count + 1):
            path = self.data_dir / f"dataset_good_{i}.npz"

            if not path.exists():
                raise FileNotFoundError(path)

            data = np.load(path)

            good_sequences.append(data["sequences"])

        bad_path = self.data_dir / self.bad_file

        if not bad_path.exists():
            raise FileNotFoundError(bad_path)

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

    def train(self):
        """
        Train posture classification model.
        """

        x, y = self.load_dataset()

        self.model.fit(x, y)

        good_samples = x[y == 0]

        self.feature_mean = np.mean(good_samples, axis=0)
        self.feature_std = np.std(good_samples, axis=0)

        return self

    def export(self, save_path="pose_model.pkl"):
        """
        Export runtime model bundle.
        """

        if self.feature_mean is None:
            raise RuntimeError(
                "Model is not trained. Call train() first."
            )

        bundle = {
            "model": self.model,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
        }

        save_path = Path(save_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(bundle, save_path)

        print(f"[INFO] Model exported -> {save_path}")

        return save_path


if __name__ == "__main__":
    trainer = PoseTrainer(
        data_dir="data/datasets",
        good_count=4,
        bad_file="dataset_bad_1.npz",
    )

    trainer.train()

    trainer.export("model/pose_model.pkl")
