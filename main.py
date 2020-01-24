import argparse

from segmentation import Segmentation, acc_camvid
#from inference import ImageObjects
#from inpainting import FeatureLoss


def main():

    #img, mask, classes = Segmentation(
    #    fname_img=args.image, model=args.model_segmentation
    #).start_segmentation()
    # ImageObjects(img, mask, classes, model=args.model_inpainting).inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image File Name")
    parser.add_argument(
        "--image", type=str, default="img.png", help="Enter the image file name"
    )
    parser.add_argument(
        "--model_segmentation",
        type=str,
        default="detection_retina_256.pth",
        help="Pre-trained Weights for RetinNet Detection",
    )
    args = parser.parse_args()

    main()
