import math
import numpy as np
from cv2 import cv2


class UI:
    "A Wrapper for In-Window Interface Variables"

    def __init__(self):
        # Click count and co-ordinates (upto 4 co-ordinates)
        self.clickCount = 0
        self.mouseCoords = []

        # Bounding Boxes For Various Objects
        self.bboxGreen = []  # Parkable Space
        self.bboxYellow = []  # Detected Vehicle
        self.bboxRed = []  # Occupied Space


class Inference:
    "A Wrapper for Inference"

    def __init__(
        self,
        # model,
        reference_img="img.png",
        path_to_input="./img_input/",
        content_img="img.png",
    ):

        self.reference_img = cv2.imread(path_to_input + content_img)
        self.img = cv2.imread(path_to_input + content_img)
        self.img_display = np.copy(self.img)

        self.ui = UI()
        # self.detection = model

        # This will be the default window for inference
        cv2.namedWindow(winname="Ocular Parking System")
        # self.bool_segment = cv2.createTrackbar(
        #     "Show Segments", "Image Objects", 0, 1, self.detection.detect_thresh
        # )
        cv2.setMouseCallback("Ocular Parking System", self.mouse_events)

    # def show(self):
    #     "Displays content input image and its corresponding detected objects"

    #     for bbox in self.bboxes:
    #         top_left = bbox[0]
    #         bottom_right = bbox[1]

    #         cv2.rectangle(
    #             self.img_display,
    #             top_left,
    #             bottom_right,
    #             color=(0, 200, 255),
    #             thickness=2,
    #         )

    #         # text = f"{classes[preds[i]]} {scores[i]}"
    #         # cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    #     cv2.imshow(winname="Inference Window", mat=self.img_display)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def parkable_region_inference(self):
        "Inference for manually defining green regions"
        print(
            "Usage:\n\t1.) Register 4 Left-clicks (Rectangle co-ordinates) to add parking regions \n\t2.) Right-click to delete a region  \n\t3.) Press 'y' to continue"
        )

        while True:
            cv2.imshow(winname="Ocular Parking System", mat=self.reference_img)
            if cv2.waitKey(60) & 0xFF == ord("y"):
                break

        cv2.destroyAllWindows()

    def mouse_to_region(self):
        " Adds 'cv2.RotatedRect' Region On Mouse-Clicked 'mouseCoords' "

        bbox = cv2.minAreaRect(np.array(self.ui.mouseCoords))
        contour = np.intp(cv2.boxPoints(bbox))

        cv2.drawContours(
            image=self.reference_img,
            contours=[contour],
            contourIdx=-1,
            color=(75, 150, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        self.ui.bboxGreen.append(contour)
        self.ui.clickCount = 0
        self.ui.mouseCoords = []

        print("\nCo-ordinates Added!\n\t{contour}")
        print(
            "\n\nUsage:\n\t1.) Register 4 Left-clicks (Rectangle co-ordinates) to add parking regions \n\t2.) Right-click to delete a region  \n\t3.) Press 'y' to continue"
        )

    def mouse_events(self, event, x, y, flags, param):
        "Mouse callback function definition"

        if event == cv2.EVENT_LBUTTONDOWN:
            "Adds co-ordinates from mouse input (Left-Click)"
            self.ui.mouseCoords.append([x, y])
            self.ui.clickCount += 1
            print(f"Click! {self.ui.clickCount}")
            cv2.circle(self.reference_img, (x, y), 2, (0, 200, 255))

            # self.obj_selected = self.contour_containing_point(x, y)
            # self.show_objects(fromClean=True)

        if event == cv2.EVENT_MOUSEMOVE:
            "Later"

        if event == cv2.EVENT_LBUTTONUP:
            "Later"

        if self.ui.clickCount >= 4:
            self.mouse_to_region()
            # Check for 4 clicks (of a rectangle) for LBUTTON event
            # self.mouse_to_region()

    def start_inference(self):
        "Mouse-Events Ready User Interface"
        self.parkable_region_inference()
