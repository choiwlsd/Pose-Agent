import cv2

from agent_model.pose_agent import PoseAgent


if __name__ == "__main__":
    source = cv2.VideoCapture(filename='./asset/input_bad.mp4')
    # source = cv2.VideoCapture(0)  # 웹캠 사용 시
    source.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    agent = PoseAgent(source)
    agent.run()

