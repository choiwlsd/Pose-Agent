from pose_extractor import PoseExtractor
from feature_extractor import FeatureExtractor


# 전역 변수로 pose_extractor와 feature_extractor 인스턴스 생성
poseExtractor = PoseExtractor()
featureExtractor = FeatureExtractor()


# pose_extractor에서 매 프레임마다 landmarks를 받아오는 콜백
# landmarks는 {idx: landmark} 형태의 딕셔너리
def on_landmarks(landmarks):

    if landmarks is None:
        return
    
    features = featureExtractor.compute(landmarks)

    if features is None:
        return
    
    # wts = features['wrist_to_shoulder']
    # sw  = features['shoulder_width']

    # if 'left' in wts:
    #     print(f"왼손목-왼어깨 거리:  {wts['left']:.3f}")
    # if 'right' in wts:
    #     print(f"오른손목-오른어깨 거리: {wts['right']:.3f}")
    # if sw:
    #     print(f"어깨 너비: {sw:.3f}")  

    print(features)


    # presence 통과한 keypoints만 출력 
    for idx, lm in landmarks.items():
        print(f"[{idx}] x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, presence={lm.presence:.2f}")

if __name__ == "__main__":
    poseExtractor.run(callback=on_landmarks)