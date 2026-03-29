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
    
    wts = features['wrist_to_shoulder']
    sw  = features['shoulder_width']
    wv  = features['wrist_velocity']
    ea  = features['elbow_angle']
    wa  = features['wrist_angle']


    if 'left' in wts:
        print(f"왼손목-왼어깨 거리:  {wts['left']:.3f}")
    if 'right' in wts:
        print(f"오른손목-오른어깨 거리: {wts['right']:.3f}")
    if sw:
        print(f"어깨 너비: {sw:.3f}")  
    if wv:
        if 'left' in wv:
            print(f"왼손목 속도: {wv['left']:.4f}")
        if 'right' in wv:
            print(f"오른손목 속도: {wv['right']:.4f}")
    if 'left' in ea:
        print(f"왼팔꿈치 각도:         {ea['left']:.1f}°")
    if 'right' in ea:
        print(f"오른팔꿈치 각도:       {ea['right']:.1f}°")
    if 'left' in wa:
        print(f"왼손목 각도:           {wa['left']:.1f}°")
    if 'right' in wa:
        print(f"오른손목 각도:         {wa['right']:.1f}°")


    # presence 통과한 keypoints만 출력 
    for idx, lm in landmarks.items():
        print(f"[{idx}] x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, presence={lm.presence:.2f}")

if __name__ == "__main__":
    poseExtractor.run(callback=on_landmarks)