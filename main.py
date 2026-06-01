import cv2

from agent_model.pose_agent import PoseAgent


if __name__ == "__main__":
    webcam = cv2.VideoCapture(filename='./asset/input_good.mp4')
    # webcam = cv2.VideoCapture(0)  # 웹캠 사용 시
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    agent = PoseAgent(webcam)
    agent.run()

