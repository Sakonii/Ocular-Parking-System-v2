import argparse

from detection import Detection
from inference import Inference
from cv2 import cv2


def main():
    detection = Detection(modelWeights=args.model_detection, cfgPath= args.cfg_path)
    Inference(
        content_video=args.video, model=detection
    ).start_inference(generateMap=True)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image File Name")
    parser.add_argument(
        "--video",
        type=str,
        default="videoplayback2.mp4",
        help="Enter the image / video file name located at ./img_input",
    )
    parser.add_argument(
        "--model_detection",
        type=str,
        default="http://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
        help="Pre-trained Weights for Detectron Detection",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="/configs/faster_rcnn_R_50_FPN_3x.yaml",
        help="Path to model cfg file relative to 'detectron2/model_zoo' ",
    )
    args = parser.parse_args()

    main()
