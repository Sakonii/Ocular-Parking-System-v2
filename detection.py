import numpy as np
from cv2 import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class Detection:
    "A Wrapper for Detection related objects"

    def __init__(self, modelWeights, cfgPath):
        # Load Detection Model
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(cfgPath))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set detection threshold
        self.cfg.MODEL.WEIGHTS = modelWeights
        self.cfg.MODEL.DEVICE = "cpu"
        self.predict = DefaultPredictor(self.cfg)

    def update_threshold(self):
        " Updates Detection Threshold On Trackbar Event"
        self.predict = DefaultPredictor(self.cfg)

    def tensor_to_np(self):
        "Convert Tensors to OpenCV Comaptible Types"
        # Convert to np.ndarray
        self.bboxes = self.outputs["instances"].pred_boxes.tensor.numpy().astype(int)
        self.preds = self.outputs["instances"].pred_classes.numpy().astype(int)
        self.scores = self.outputs["instances"].scores.numpy().astype(int)
        # Convert to list (if not already)
        if not isinstance(self.bboxes, list):
            self.bboxes = self.bboxes.tolist()
            self.preds = self.preds.tolist()
            self.scores = self.scores.tolist()
        # Convert bboxes to list of top-left and bottom right co-ordinates
        _bboxes = []
        for _bbox in self.bboxes:
            top_left = (_bbox[0], _bbox[1])
            bottom_right = (_bbox[2], _bbox[3])
            _bboxes.append([top_left, bottom_right])
        self.bboxes = _bboxes

    def show(self):
        "Displays content input image and its corresponding detected objects"
        for bbox in self.bboxes:
            top_left = bbox[0]
            bottom_right = bbox[1]
            # Default Color: Green
            cv2.rectangle(
                self.img_display,
                top_left,
                bottom_right,
                color=(0, 200, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            # text = f"{classes[preds[i]]} {scores[i]}"
            # cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow(winname="Ocular Parking System", mat=self.img_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_detections(self, img):
        "Displays content input image and its corresponding detected torch objects"
        v = Visualizer(
            img[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0
        )
        v = v.draw_instance_predictions(self.outputs["instances"])
        return v.get_image()[:, :, ::-1]

    def start_detection(self, img):
        "Updates and Returns bboxes, preds, scores and classes for next video frame"
        self.img = img
        self.img_display = np.copy(self.img)
        # Model Prediction
        self.outputs = self.predict(img)
        self.tensor_to_np()
        # self.show()
        return (self.bboxes, self.preds, self.scores)
