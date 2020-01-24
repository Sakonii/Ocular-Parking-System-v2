import argparse

from detection import Detection
#from inference import ImageObjects
#from inpainting import FeatureLoss


def main():

    Detection(
       fname_img=args.image, model_pth=args.model_detection
    ).start_detection()
    # ImageObjects(img, mask, classes, model=args.model_inpainting).inference()
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
