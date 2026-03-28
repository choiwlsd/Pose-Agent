from pose_extractor import PoseExtractor

# pose_extractor에서 매 프레임마다 landmarks를 받아오는 콜백
# landmarks는 {idx: landmark} 형태의 딕셔너리
def on_landmarks(landmarks):

    if landmarks is None:
        return
    
    # presence 통과한 keypoints만 출력 
    for idx, lm in landmarks.items():
        print(f"[{idx}] x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, presence={lm.presence:.2f}")

if __name__ == "__main__":
    extractor = PoseExtractor()
    extractor.run(callback=on_landmarks)