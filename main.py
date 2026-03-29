from pose_extractor import PoseExtractor
from feature_extractor import FeatureExtractor
from visualizer import print_features


# 전역 변수로 pose_extractor와 feature_extractor 인스턴스 생성
poseExtractor = PoseExtractor()
featureExtractor = FeatureExtractor()


# pose_extractor에서 매 프레임마다 landmarks를 받아오는 콜백
# landmarks는 {idx: landmark} 형태의 딕셔너리
def on_landmarks(landmarks):

    if landmarks is None:
        return
    
    features = featureExtractor.compute(landmarks)
    print_features(features)

    # TCN 입력용: feature vector로 변환 
    feature_vector = featureExtractor.to_vector(features)
    print(f"Feature Vector: {feature_vector}")

    # TCN 입력용 시퀀스: feature vector를 시퀀스 버퍼에 추가 
    sequence = featureExtractor.update_buffer(features)
    print(sequence.shape if sequence is not None else None)

    # presence 통과한 keypoints만 출력 
    for idx, lm in landmarks.items():
        print(f"[{idx}] x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, presence={lm.presence:.2f}")

if __name__ == "__main__":
    poseExtractor.run(callback=on_landmarks)