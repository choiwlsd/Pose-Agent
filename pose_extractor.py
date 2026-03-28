import cv2
import mediapipe as mp
from collections import deque

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# 연결선 정의 (어떤 keypoint끼리 이을지)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

class PoseExtractor:
    def __init__(self, model_path='pose_landmarker_lite.task', presence_threshold=0.5):
        self.presence_threshold = presence_threshold

        # 결과 저장용
        self.latest_result = None

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._on_result
        )

        self.landmarker = PoseLandmarker.create_from_options(options)
        
        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)   
        
        # 버퍼를 1로 설정: 최신 프레임만 유지
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   
        
        # 프레임 순서 카운터
        self.timestamp = 0   

    # MediaPipe 결과 콜백: 결과를 latest_result에 저장
    def _on_result(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_result = result

    # presence 통과한 keypoints만 반환
    def get_landmarks(self):
        if not self.latest_result or not self.latest_result.pose_landmarks:
            return None
        landmarks = self.latest_result.pose_landmarks[0] # 첫 번째 사람의 keypoints
        return {idx: lm for idx, lm in enumerate(landmarks) if lm.presence >= self.presence_threshold}

    # keypoints 및 연결선 시각화
    def draw(self, frame, landmarks):
        if not landmarks:
            return frame
        
        h, w = frame.shape[:2]
        points = {}

        # presence 통과한 keypoints만 그리기
        for idx, lm in landmarks.items():
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[idx] = (cx, cy)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(idx), (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
        # 연결선 그리기    
        for a, b in CONNECTIONS:
            if a in points and b in points:
                cv2.line(frame, points[a], points[b], (0, 180, 255), 2)

        return frame

    # 메인 루프: 프레임 읽고, landmarks 추출, callback으로 landmarks main.py에 전달 
    def run(self, callback=None):
        while self.cap.isOpened():
            ret, frame = self.cap.read() # 프레임 1장 읽기: ret은 성공 여부, frame은 이미지 데이터
            if not ret:
                break

            # 프레임을 RGB MediaPipe Image로 변환
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            self.timestamp += 1
            self.landmarker.detect_async(mp_image, self.timestamp)

            # presence 통과한 keypoints만 추출
            landmarks = self.get_landmarks()

            # main.py로 landmarks 전달
            if callback:
                callback(landmarks)

            frame = self.draw(frame, landmarks)
            cv2.imshow("MediaPipe Pose", frame)

            # 1ms 대기 후 'q' 키 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release()

    # 웹캠과 MediaPipe 리소스 해제
    def release(self):
        self.cap.release()
        self.landmarker.close()
        cv2.destroyAllWindows()