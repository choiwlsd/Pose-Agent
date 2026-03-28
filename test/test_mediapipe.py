import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# 연결선 정의 (어떤 keypoint끼리 이을지)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),           # 코-눈
    (0, 4), (4, 5), (5, 6),           # 코-눈(반대)
    (9, 10),                          # 입
    (11, 12),                         # 어깨
    (11, 13), (13, 15),               # 왼팔
    (12, 14), (14, 16),               # 오른팔
    (11, 23), (12, 24), (23, 24),     # 몸통
    (23, 25), (25, 27),               # 왼다리
    (24, 26), (26, 28),               # 오른다리
]

# 결과 저장용
latest_result = None

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)



# 웹캠 열기 + 영상 읽기
cap = cv2.VideoCapture(0)
# 타임스탬프 초기화 (실제 시간 대신 프레임 단위로 증가시키는 방식)
timestamp = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    # 웹 캠이 열려있는 동안 반복 
    while cap.isOpened():
        ret, frame = cap.read()  # 프레임 1장 읽기: ret은 성공 여부, frame은 이미지 데이터
        if not ret:
            break   # 프레임을 제대로 읽지 못했으면 종료

        # MediaPipe Image로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 추론
        timestamp += 1
        landmarker.detect_async(mp_image, timestamp)

        # Keypoints 출력
        if latest_result and latest_result.pose_landmarks:
            landmarks = latest_result.pose_landmarks[0]

            # 시각화 - 연결선 그리기
            points = {}
            for idx, lm in enumerate(landmarks):
                if lm.presence < 0.5:
                    continue
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                points[idx] = (x, y)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            for a, b in CONNECTIONS:
                if a in points and b in points:
                    cv2.line(frame, points[a], points[b], (0, 255, 255), 2)

            # 디버그용으로 keypoint 정보 출력
            for idx, lm in enumerate(landmarks):
                if lm.presence < 0.5:
                    print(f"[{idx}] Skipped due to low presence: {lm.presence:.2f}")
                    continue
                print(f"[{idx}] x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, presence={lm.presence:.2f}")

        cv2.imshow("MediaPipe Pose", frame)
        
        # 1ms마다 'q' 키 입력 대기 -> 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):         
            break

# 웹캠과 창 닫기
cap.release()
cv2.destroyAllWindows()
