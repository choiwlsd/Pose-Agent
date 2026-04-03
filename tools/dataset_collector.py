import os
import numpy as np
from model.pose_extractor import PoseExtractor
from model.feature_extractor import FeatureExtractor

LABELS = {
    0: 'good',
    1: 'bad',
}
LABEL_TO_IDX = {v: k for k, v in LABELS.items()} # {'good': 0, 'bad': 1}

class DatasetCollector:
    def __init__(self):
        self.sequences = []  # (N, 30, 6) feature sequences
        self.labels = []     # (N,) (0: good, 1: bad)

    def process_video(self, video_path, label):
        # 영상 1개 처리 → (30, 6) sequence 추출 후 바로 저장
        featureExtractor = FeatureExtractor() # 영상마다 버퍼 초기화
        poseExtractor = PoseExtractor(source=video_path)

        def on_landmarks(landmarks):
            features = featureExtractor.compute(landmarks)
            sequence = featureExtractor.update_buffer(features)
            if sequence is not None:
                self.sequences.append(sequence)
                self.labels.append(label)

        poseExtractor.run(callback=on_landmarks, display=False) # display=False: 영상 출력 없이 처리
        print(f"  완료: {os.path.basename(video_path)} → {len(self.sequences)}개 sequence 누적")


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

        print(f"\n[{label_name}] {folder_path} ({len(video_files)}개 영상)")
        for filename in video_files:
            video_path = os.path.join(folder_path, filename)
            print(f"  처리 중: {filename}")
            self.process_video(video_path, label)

        # 개별 저장
        labels_np    = np.array(self.labels,    dtype=np.int64)
        sequences_np = np.array(self.sequences, dtype=np.float32)
        mask         = labels_np == label

        os.makedirs('data/datasets', exist_ok=True)
        save_path = f'data/datasets/dataset_{label_name}.npz'
        np.savez(save_path, sequences=sequences_np[mask], labels=labels_np[mask])
        print(f"  개별 저장 완료: {save_path} → {sequences_np[mask].shape}")


    def get_stats(self):
        if not self.sequences:
            print("수집된 데이터가 없어요!")
            return
        labels = np.array(self.labels)
        print(f"\n=== 데이터 수집 현황 ===")
        print(f"전체 sequences: {len(self.sequences)}개")
        print(f"good (0): {(labels == 0).sum()}개")
        print(f"bad  (1): {(labels == 1).sum()}개")


    def save(self, save_path='data/datasets/dataset.npz'):
        if not self.sequences:
            print("저장할 데이터가 없어요!")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        sequences = np.array(self.sequences, dtype=np.float32)  # (N, 30, 6)
        labels    = np.array(self.labels,    dtype=np.int64)     # (N,)

        np.savez(save_path, sequences=sequences, labels=labels)

        self.get_stats()
        print(f"\n저장 완료: {save_path}")
        print(f"sequences shape: {sequences.shape}")

if __name__ == "__main__":
    collector = DatasetCollector()
    collector.process_folder('data/raw/good', label='good')
    collector.process_folder('data/raw/bad',  label='bad')
    collector.save()
