import math
import numpy as np
from cv2 import cv2
from map import Map


class UI:
    "A Wrapper for In-Window Interface Variables"

    def __init__(self):
        # Click count and co-ordinates (upto 4 co-ordinates)
        self.clickCount = 0
        self.mouseCoords = []
        self.selectedContour = np.intp()
        # Bounding Boxes For Various Objects
        self.bboxesGreen = []  # Parkable Space
        self.bboxesRed = []  # Occupied Space
        # self.detection.bboxes = []  # Detected Vehicle

    def reset_mouse_coords(self):
        self.clickCount = 0
        self.mouseCoords = []


class Inference:
    "A Wrapper for Inference"

    def __init__(
        self, model, path_to_input="./img_input/", content_video="img.png",
    ):
        self.cap = cv2.VideoCapture(path_to_input + content_video)
        _, self.reference_img = self.cap.read()  # First Frame OF Video
        self.img = None
        self.img_display = None
        # Instantiate / Import UI Components and Detection Model
        self.ui = UI()
        self.detection = model
        self.map = Map()
        # Frame Counter For FPS
        self.frameCounter = 0
        # This will be the default window for inference
        cv2.namedWindow(winname="Ocular Parking System")
        # Trackbar Callbacks
        cv2.createTrackbar(
            "Reservation Mode", "Ocular Parking System", 0, 1, lambda *_, **__: None
        )
        cv2.createTrackbar(
            "Detect Threshold",
            "Ocular Parking System",
            60,
            100,
            self.detection.update_threshold,
        )
        cv2.createTrackbar(
            "Frames / Detection", "Ocular Parking System", 3, 10, lambda *_, **__: None
        )

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
            # Default Color: Yellow
            cv2.rectangle(
                img,
                top_left,
                bottom_right,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    def contour_containing_point(self, x, y, contours, boolDebug=False, bool=False):
        "Returns contour under which the co-ordinate lies"
        for contour in contours:
            if cv2.pointPolygonTest(contour=contour, pt=(x, y), measureDist=False) >= 0:
                if boolDebug:
                    print(contour)
                return True if bool else contour
            else:
                pass
        return False if bool else []

    def remove_contour_containing_point(self, x, y, contours, boolDebug=False):
        "Removes a detected 'contour' region from co-ordinate x y"
        for contour in contours:
            if cv2.pointPolygonTest(contour=contour, pt=(x, y), measureDist=False) >= 0:
                if boolDebug:
                    print(f"Removed:\n{contour}")
                self.remove_np_array_from_list(contours, contour)
            else:
                pass

    def generate_points_to_check(self, bbox, boolDebug=False):
        "Generates list of co-ordinates from bottom-center to center of bbox/rectangle"
        center_x = int(
            bbox[1][0] - (bbox[1][0] - bbox[0][0]) / 2
        )  # center = x2 - (x2-x1)/2
        center_y = int(bbox[1][1] - (bbox[1][1] - bbox[0][1]) / 2)
        center_y_bottom = bbox[1][1]
        # Generate 10 linspaced y-coordinates
        ysToCheck = np.linspace(center_y_bottom, center_y, num=8, dtype=int)
        pointsToCheck = []
        for yToCheck in ysToCheck:
            pointsToCheck.append([center_x, yToCheck])
            pointsToCheck.append(
                [center_x - int((center_x - bbox[0][0]) / 4), yToCheck]
            )
            pointsToCheck.append(
                [center_x + int((center_x - bbox[0][0]) / 4), yToCheck]
            )
        # Print Co-ordinates
        if boolDebug:
            print(f"Center: {center_x, center_y}")
            print(f"Center Bottom: {center_x, center_y_bottom}")
            print(f"Generated: {pointsToCheck}")
        return pointsToCheck

    def remove_np_array_from_list(self, listOfArray, npArray):
        # Custom List Item Removal Function
        ind = 0
        size = len(listOfArray)
        while ind != size and not np.array_equal(listOfArray[ind], npArray):
            ind += 1
        if ind != size:
            listOfArray.pop(ind)
        else:
            raise ValueError("array not found in list.")

    def occupied_space_detection(self):
        "Updates List Of Red/Occupied/bboxRed Regions On Obstruction Detection"
        if not self.ticketMode:
            for bboxRed in self.ui.bboxesRed:
                boolRedInNewFrame = False
                for bboxYellow in self.detection.bboxes:
                    pointsToCheck = self.generate_points_to_check(bboxYellow)
                    for point in pointsToCheck:  # for points: center to bottom-center
                        if self.contour_containing_point(
                            point[0], point[1], [bboxRed], bool=True
                        ):
                            boolRedInNewFrame = True
                            break
                    else:
                        continue
                    break
                if not boolRedInNewFrame:
                    self.remove_np_array_from_list(self.ui.bboxesRed, bboxRed)

        for bboxYellow in self.detection.bboxes:  # for every bboxYellow
            pointsToCheck = self.generate_points_to_check(bboxYellow)
            for bboxGreen in self.ui.bboxesGreen:  # for every bboxGreen
                for point in pointsToCheck:  # for points: center to bottom-center
                    if self.contour_containing_point(  # if point lies in bboxGreen
                        point[0], point[1], [bboxGreen], bool=True
                    ):
                        if not np.any(  # if bboxGreen not in bboxesRed
                            bboxGreen == self.ui.bboxesRed
                        ):
                            self.ui.bboxesRed.append(bboxGreen)
                        break
                else:
                    continue
                break

    def video_inference(self, generateMap=True):
        "Inference For The Main Program"
        _, self.img = self.cap.read()
        self.img_display = np.copy(self.img)
        # Trackbar Callbacks
        self.ticketMode = cv2.getTrackbarPos(
            "Reservation Mode", "Ocular Parking System"
        )
        self.updateTimeSec = cv2.getTrackbarPos(
            "Frames / Detection", "Ocular Parking System"
        )
        self.detection.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
            cv2.getTrackbarPos("Detect Threshold", "Ocular Parking System")
        ) / 100
        # Frame Skipping
        if (self.frameCounter == 0) | (
            self.frameCounter % (30 * self.updateTimeSec) == 0
        ):
            self.detection.start_detection(self.img)
            self.occupied_space_detection()
            if generateMap:
                availableSpots = len(self.ui.bboxesGreen) - len(self.ui.bboxesRed)
                self.map.locationsData.emptySpots[3] = availableSpots
                self.map.generate()
        # Draw Green Parkable Regions
        self.draw_contours(
            contours=self.ui.bboxesGreen, img=self.img_display, color=(75, 150, 0),
        )
        # Draw Red Occupied Regions
        self.draw_contours(
            contours=self.ui.bboxesRed, img=self.img_display, color=(0, 50, 255),
        )
        # Draw Yellow Detected Bboxes
        if (self.frameCounter == 0) | (
            self.frameCounter % (30 * self.updateTimeSec) <= 40
        ):
            # self.draw_bboxes(self.img_display, color=(0, 200, 255))
            self.img_display = self.detection.draw_detections(self.img_display)
        self.frameCounter += 1
        # Print Occupancy Information
        print(f"\n\n\n\nTotal parking Spots   = {len(self.ui.bboxesGreen)}")
        print(f"No. of Spots Occupied = {len(self.ui.bboxesRed)}")
        print(
            f"Total available Spots = {len(self.ui.bboxesGreen) - len(self.ui.bboxesRed)}"
        )
        print(f"\nMiddle-Mouse Click to reserve a space")
        # Park-able Region Detection Window
        cv2.imshow(winname="Ocular Parking System", mat=self.img_display)
        if cv2.waitKey(60) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            quit()

    def parkable_region_inference(self):
        "Inference For Manually Calibrating Green Regions"
        print(
            "Usage:\n\t1.) Register 4 Left-clicks (Rectangle co-ordinates) to add parking regions"
            "\n\t2.) Right-click to delete a region  \n\t3.) Press 'y' to continue"
        )
        # Calibration Inference
        while True:
            self.reference_img_display = np.copy(self.reference_img)
            # Draw Green Parkable Regions
            self.draw_contours(
                contours=self.ui.bboxesGreen,
                img=self.reference_img_display,
                color=(75, 150, 0),
            )
            # Park-able Region Detection Window
            cv2.imshow(winname="Ocular Parking System", mat=self.reference_img_display)
            if cv2.waitKey(60) & 0xFF == ord("y"):
                break
        # cv2.destroyAllWindows()

    def mouse_to_region(self):
        " Adds 'cv2.RotatedRect' Region On Mouse-Clicked 'mouseCoords' "
        # Convert Python Lists to OpenCV compatible boxPoints
        bbox = cv2.minAreaRect(np.array(self.ui.mouseCoords))
        self.ui.selectedContour = np.intp(cv2.boxPoints(bbox))
        # Append To Original UI List
        self.ui.bboxesGreen.append(self.ui.selectedContour)
        # Reset Mouse Click Counter
        self.ui.reset_mouse_coords()

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
            self.remove_contour_containing_point(x, y, self.ui.bboxesGreen)
            self.remove_contour_containing_point(x, y, self.ui.bboxesRed)

        if event == cv2.EVENT_MBUTTONDOWN:
            "Reserve a parking space"
            self.ui.reset_mouse_coords()
            tempContour = self.contour_containing_point(x, y, self.ui.bboxesGreen)
            if tempContour == []:
                pass
            elif self.contour_containing_point(x, y, self.ui.bboxesRed, bool=True):
                self.remove_np_array_from_list(self.ui.bboxesRed, tempContour)
            else:
                self.ui.bboxesRed.append(tempContour)

        if self.ui.clickCount >= 4:
            # Check for 4 clicks (of a rectangle) for LBUTTON event
            self.mouse_to_region()

    def start_inference(self):
        "Mouse-Events Ready User Interface"
        self.parkable_region_inference()
        while self.cap.isOpened():
            self.video_inference()
