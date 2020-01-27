import math
import numpy as np
from cv2 import cv2


class UI:
    "A Wrapper for In-Window Interface Variables"

    def __init__(self):
        # Click count and co-ordinates (upto 4 co-ordinates)
        self.clickCount = 0
        self.mouseCoords = []
        self.selectedContour = np.intp()
        # Bounding Boxes For Various Objects
        self.bboxGreen = []  # Parkable Space
        self.bboxYellow = []  # Detected Vehicle
        self.bboxRed = []  # Occupied Space


class Inference:
    "A Wrapper for Inference"

    def __init__(
        self,
        model,
        reference_img="img.png",
        path_to_input="./img_input/",
        content_img="img.png",
    ):
        self.reference_img = cv2.imread(path_to_input + content_img)
        self.img = cv2.imread(path_to_input + content_img)
        self.img_display = np.copy(self.img)
        # Instantiate / Import UI Components and Detection Model
        self.ui = UI()
        self.detection = model
        # This will be the default window for inference
        cv2.namedWindow(winname="Ocular Parking System")
        # self.bool_segment = cv2.createTrackbar(
        #     "Show Segments", "Image Objects", 0, 1, self.detection.detect_thresh
        # )
        cv2.setMouseCallback("Ocular Parking System", self.mouse_events)

    def draw_contours(self, contours, img, color=(75, 150, 0)):
        "Draws list of contours"
        cv2.drawContours(
            image=img,
            contours=contours,
            contourIdx=-1,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    def draw_bboxes(self, img, color=(0, 200, 255)):
        "Draws rectangle from top_left and bottom_right co-ordinates at self.detection.bboxes"
        for bbox in self.detection.bboxes:
            top_left = bbox[0]
            bottom_right = bbox[1]
            # Default Color: Green
            cv2.rectangle(
                img,
                top_left,
                bottom_right,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            # cv2.circle(self.img_display, top_left, 2, (255, 0, 0), thickness=-1)
            # cv2.circle(self.img_display, bottom_right, 2, (255, 0, 0), thickness=-1)

            # text = f"{classes[preds[i]]} {scores[i]}"
            # cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    def contour_containing_point(self, x, y, contours, boolDebug=False):
        "Returns the contour under which the co-ordinate falls"
        for contour in contours:
            if (
                cv2.pointPolygonTest(contour=contour, pt=(x, y), measureDist=False) >= 0
                and contour is not self.ui.selectedContour
            ):
                if boolDebug:
                    print(f"Removed:\n{contour}")
                return contour
            else:
                pass
        return []

    def remove_contour_containing_point(self, x, y, contours, boolDebug=False):
        "Removes a detected 'contour' region from co-ordinate x y"
        for contour in contours:
            if cv2.pointPolygonTest(contour=contour, pt=(x, y), measureDist=False) >= 0:
                if boolDebug:
                    print(contour)
                contours.remove(contour)
            else:
                pass

    def parkable_region_inference(self):
        "Inference for manually defining green regions"
        print(
            "Usage:\n\t1.) Register 4 Left-clicks (Rectangle co-ordinates) to add parking regions"
            "\n\t2.) Right-click to delete a region  \n\t3.) Press 'y' to continue"
        )

        while True:
            self.reference_img_display = np.copy(self.reference_img)
            # Draw Green Parkable Regions
            self.draw_contours(
                contours=self.ui.bboxGreen,
                img=self.reference_img_display,
                color=(75, 150, 0),
            )
            # Draw Yellow Detected Bboxes
            self.draw_bboxes(self.reference_img_display, color=(0, 200, 255))
            # Park-able Region Detection Window
            cv2.imshow(winname="Ocular Parking System", mat=self.reference_img_display)
            if cv2.waitKey(60) & 0xFF == ord("y"):
                break

        cv2.destroyAllWindows()

    def mouse_to_region(self):
        " Adds 'cv2.RotatedRect' Region On Mouse-Clicked 'mouseCoords' "
        # Convert Python Lists to OpenCV compatible boxPoints
        bbox = cv2.minAreaRect(np.array(self.ui.mouseCoords))
        self.ui.selectedContour = np.intp(cv2.boxPoints(bbox))
        # Append To Original UI List
        self.ui.bboxGreen.append(self.ui.selectedContour)
        # Reset Mouse Click Counter
        self.ui.clickCount = 0
        self.ui.mouseCoords = []

        print(f"\nCo-ordinates Added!\n{self.ui.selectedContour}")
        print(
            "\n\nUsage:\n\t1.) Register 4 Left-clicks (Rectangle co-ordinates) to add parking regions"
            "\n\t2.) Right-click to delete a region  \n\t3.) Press 'y' to continue"
        )

    def mouse_events(self, event, x, y, flags, param):
        "Mouse callback function definition"

        if event == cv2.EVENT_LBUTTONDOWN:
            "Adds co-ordinates from mouse input (Left-Click)"
            self.ui.mouseCoords.append([x, y])
            self.ui.clickCount += 1
            print(f"Click! {self.ui.clickCount}")
            cv2.circle(self.reference_img, (x, y), 2, (0, 200, 255))

        if event == cv2.EVENT_RBUTTONDOWN:
            "Removes a detected 'contour' region from co-ordinate x y"
            self.remove_contour_containing_point(x, y, self.ui.bboxGreen)
            self.remove_contour_containing_point(x, y, self.ui.bboxRed)

        if event == cv2.EVENT_LBUTTONUP:
            "Later"

        if self.ui.clickCount >= 4:
            # Check for 4 clicks (of a rectangle) for LBUTTON event
            self.mouse_to_region()

    def start_inference(self):
        "Mouse-Events Ready User Interface"
        self.parkable_region_inference()
