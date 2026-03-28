import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    # 두 landmark 사이의 2D 유클리드 거리 계산
    def _distance(self, a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

    # 손목-어깨 거리 계산
    # 왼손목(15)-왼어깨(11), 오른손목(16)-오른어깨(12)
    # 반환: {'left': float, 'right': float}
    def get_wrist_to_shoulder(self, landmarks):
        result = {}

        if 15 in landmarks and 11 in landmarks:
            result['left'] = self._distance(landmarks[15], landmarks[11])

        if 16 in landmarks and 12 in landmarks:
            result['right'] = self._distance(landmarks[16], landmarks[12])

        return result

    # 양쪽 어깨 사이 거리 계산
    # 왼어깨(11)-오른어깨(12)
    # 반환: float 또는 None
    def get_shoulder_width(self, landmarks):
        if 11 in landmarks and 12 in landmarks:
            return self._distance(landmarks[11], landmarks[12])
        
        return None


    # 모든 거리 feature를 한번에 계산해서 dictionary로 반환
    def compute(self, landmarks):
        if landmarks is None:
            return None

        return {
            'wrist_to_shoulder': self.get_wrist_to_shoulder(landmarks),
            'shoulder_width':    self.get_shoulder_width(landmarks),
        }