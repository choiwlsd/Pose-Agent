import os
import numpy as np
from pose_extractor import PoseExtractor
from feature_extractor import FeatureExtractor
from labels import LABEL_TO_IDX

class DatasetCollector:
    def __init__(self):
        self.sequences = []  # (N, 30, 9) feature sequences
        self.labels = []     # (N,) 라벨 (0: good, 1: bad)

    def process_video(self, video_path, label):
        # 영상 1개 처리 → (30, 9) sequence 추출
        # label: int (0 or 1)
        featureExtractor = FeatureExtractor()  # 영상마다 버퍼 초기화
        poseExtractor = PoseExtractor(source=video_path, display=False)

        def on_landmarks(landmarks):
            features = featureExtractor.compute(landmarks)
            sequence = featureExtractor.sequence_buffer(features)
            if sequence is not None:
                self.sequences.append(sequence)
                self.labels.append(label)

        poseExtractor.run(callback=on_landmarks)
        print(f"  완료: {os.path.basename(video_path)} → {len(self.sequences)}개 sequence 누적")

    def process_folder(self, folder_path, label):
        # 폴더 안의 모든 영상 처리
        # label: str (예: 'good') 또는 int (예: 0)
        if isinstance(label, str):
            label = LABEL_TO_IDX[label]

        video_files = [f for f in os.listdir(folder_path)
                       if f.endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            print(f"경고: {folder_path}에 영상 파일이 없어요!")
            return

        print(f"\n[{'good' if label == 0 else 'bad'}] {folder_path} ({len(video_files)}개 영상)")
        for filename in video_files:
            video_path = os.path.join(folder_path, filename)
            print(f"  처리 중: {filename}")
            self.process_video(video_path, label)

    def get_stats(self):
        # 수집된 데이터 현황 출력
        if not self.sequences:
            print("수집된 데이터가 없어요!")
            return
        labels = np.array(self.labels)
        print(f"\n=== 데이터 수집 현황 ===")
        print(f"전체 sequences: {len(self.sequences)}개")
        print(f"good (0): {(labels == 0).sum()}개")
        print(f"bad  (1): {(labels == 1).sum()}개")

    def save(self, save_path='dataset.npz'):
        # 수집한 데이터 저장
        if not self.sequences:
            print("저장할 데이터가 없어요!")
            return

        sequences = np.array(self.sequences, dtype=np.float32)  # (N, 30, 9)
        labels    = np.array(self.labels,    dtype=np.int64)     # (N,)

        np.savez(save_path, sequences=sequences, labels=labels)

        self.get_stats()
        print(f"\n저장 완료: {save_path}")
        print(f"sequences shape: {sequences.shape}")

if __name__ == "__main__":
    collector = DatasetCollector()
    collector.process_folder('data/good', label='good')
    collector.process_folder('data/bad',  label='bad')
    collector.save('dataset.npz')
