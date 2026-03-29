import numpy as np
from collections import deque

class FeatureExtractor:
    def __init__(self):
        self.prev_landmarks = deque(maxlen=2)  # 현재 + 이전 프레임만 유지

    def _distance(self, a, b):
        # 두 landmark 사이의 2D 유클리드 거리 계산
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

   
    def get_wrist_to_shoulder(self, landmarks):
        # 손목-어깨 거리 계산
        # 왼손목(15)-왼어깨(11), 오른손목(16)-오른어깨(12)
        # 반환: {'left': float, 'right': float}

        result = {}

        if 15 in landmarks and 11 in landmarks:
            result['left'] = self._distance(landmarks[15], landmarks[11])

        if 16 in landmarks and 12 in landmarks:
            result['right'] = self._distance(landmarks[16], landmarks[12])

        return result


    def get_shoulder_width(self, landmarks):
        # 양쪽 어깨 사이 거리 계산
        # 왼어깨(11)-오른어깨(12)
        # 반환: float 또는 None

        if 11 in landmarks and 12 in landmarks:
            return self._distance(landmarks[11], landmarks[12])
        
        return None

    def get_wrist_velocity(self, landmarks):
        # 손목의 프레임 간 이동 거리 계산 (속도 유사)
        # 왼손목(15), 오른손목(16)
        # 반환: {'left': float, 'right': float}

        self.prev_landmarks.append(landmarks)

        if len(self.prev_landmarks) < 2:
            return None  # 이전 프레임이 없으면 계산 불가
        
        prev = self.prev_landmarks[0]
        curr = self.prev_landmarks[1]

        result = {}
        if 15 in prev and 15 in curr:
            result['left'] = self._distance(prev[15], curr[15])
        if 16 in prev and 16 in curr:
            result['right'] = self._distance(prev[16], curr[16])

        return result


    def compute(self, landmarks):
        # 모든 거리 feature를 한번에 계산해서 dictionary로 반환

        if landmarks is None:
            return None

        return {
            'wrist_to_shoulder': self.get_wrist_to_shoulder(landmarks),
            'shoulder_width':    self.get_shoulder_width(landmarks),
            'wrist_velocity':    self.get_wrist_velocity(landmarks)
        }