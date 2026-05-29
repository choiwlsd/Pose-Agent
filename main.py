import cv2

from agent_model.pose_agent import PoseAgent


if __name__ == "__main__":
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    agent = PoseAgent(webcam)
    agent.run()
