import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 상위 폴더 경로 추가
from agent_model.pose_extractor import PoseExtractor
from agent_model.feature_extractor import FeatureExtractor

LABELS = {
    0: 'good',
    1: 'bad',
}
LABEL_TO_IDX = {v: k for k, v in LABELS.items()} # {'good': 0, 'bad': 1}

class DatasetCollector:
    def __init__(self):
        self.sequences = []  # (N, 30, 6) feature sequences
        self.labels = []     # (N,) (0: good, 1: bad)

    def process_video(self, video_path, label, save_path):
        sequences = []
        featureExtractor = FeatureExtractor()

        cap = cv2.VideoCapture(video_path)
        poseExtractor = PoseExtractor(source=cap)

        def on_landmarks(landmarks):
            if landmarks is None:
                return

            features = featureExtractor.compute(landmarks)
            sequence = featureExtractor.update_buffer(features)

            if sequence is not None:
                sequences.append(sequence)

        poseExtractor.run(frame_source=cap, callback=on_landmarks, display=False)

        cap.release()
        poseExtractor.release()

        if not sequences:
            print(f"  경고: sequence 없음 → {video_path}")
            return

        sequences_np = np.array(sequences, dtype=np.float32)
        labels_np = np.full(len(sequences), label, dtype=np.int64)

        np.savez(save_path, sequences=sequences_np, labels=labels_np)

        print(f"완료: {video_path} → {sequences_np.shape}")

    def process_folder(self, folder_path, label):
        if isinstance(label, str):
            label = LABEL_TO_IDX[label]

        label_name = 'good' if label == 0 else 'bad'

        # 폴더 안의 모든 영상을 각각 개별 파일로 저장
        video_files = sorted([f for f in os.listdir(folder_path)
                               if f.endswith(('.mp4', '.avi', '.mov'))])

        if not video_files:
            print(f"경고: {folder_path}에 영상 파일이 없어요!")
            return

        save_dir = os.path.join('data', 'datasets')
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n[{label_name}] {folder_path} ({len(video_files)}개 영상)")
        for i, filename in enumerate(video_files, start=1):
            video_path = os.path.join(folder_path, filename)
            save_path  = os.path.join(save_dir, f"dataset_{label_name}_{i}.npz")
            print(f"  처리 중: {filename}")
            self.process_video(video_path, label, save_path)

if __name__ == "__main__":
    collector = DatasetCollector()
    collector.process_folder('data/raw/good', label='good')
    collector.process_folder('data/raw/bad',  label='bad')
