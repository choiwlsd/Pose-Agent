import cv2
from pathlib import Path

from agent_model.pose_agent import PoseAgent

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "asset" / "input_bad.mp4"


if __name__ == "__main__":
    # source = cv2.VideoCapture(filename=str(DEFAULT_VIDEO_PATH))
    source = cv2.VideoCapture(0)  # 웹캠 사용 시
    source.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    agent = PoseAgent(source)
    agent.run()
