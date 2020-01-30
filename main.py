import argparse

from detection import Detection
from inference import Inference
from cv2 import cv2


def main():
    detection = Detection(model_pth=args.model_detection)
    Inference(
        reference_img=args.video, content_img=args.video, model=detection
    ).start_inference()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image File Name")
    parser.add_argument(
        "--video",
        type=str,
        default="videoplayback2.mp4",
        help="Enter the reference image file name located at ./img_input",
    )
    parser.add_argument(
        "--model_detection",
        type=str,
        default="detection_retina_256.pth",
        help="Pre-trained Weights for RetinNet Detection",
    )
    args = parser.parse_args()

    main()
