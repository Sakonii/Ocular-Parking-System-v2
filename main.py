import argparse

from detection import Detection
#from inference import Inference
from cv2 import cv2
# from inpainting import FeatureLoss


def main():
    reference_img = cv2.imread('./img_input/' + args.image)
    detection = Detection(model_pth=args.model_detection)
    detection.start_detection(img=reference_img)
    #Inference(reference_img=args.image, detection_obj=detection).start_inference()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image File Name")
    parser.add_argument(
        "--image", type=str, default="img.png", help="Enter the image file name"
    )
    parser.add_argument(
        "--model_detection",
        type=str,
        default="detection_retina_256.pth",
        help="Pre-trained Weights for RetinNet Detection",
    )
    args = parser.parse_args()

    main()
