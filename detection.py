import numpy as np
from cv2 import cv2

from fastai.vision import (
    open_image,
    image2np,
    defaults,
    create_body,
    models,
    torch,
    pil2tensor,
    Image,
)
from functions.model import RetinaNet, process_output, nms


class Detection:
    "A Wrapper for Detection related objects"

    def __init__(
        self, model_pth, path_to_weights="./models/",
    ):
        self.model_path = path_to_weights + model_pth
        # Load Detection Model
        defaults.device = torch.device("cpu")
        self.encoder = create_body(models.resnet50, cut=-2)
        self.model = RetinaNet(self.encoder, 21, final_bias=-4)
        self.state_dict = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(self.state_dict["model"], strict=False)
        self.model.eval()
        # Prediction Variables Initialization
        self.model_pred = None
        self.bboxes = None
        self.scores = None
        self.preds = None
        # List of Prediction Classes
        self.classes = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    def supress_outputs(self):
        "Apply Non-Max Supression"
        if len(self.preds) != 0:
            "If any detection"
            to_keep = nms(self.bboxes, self.scores)
            self.bboxes, self.preds, self.scores = (
                self.bboxes[to_keep].cpu(),
                self.preds[to_keep].cpu(),
                self.scores[to_keep].cpu(),
            )
            t_sz = torch.Tensor([*self.img_model.size])[None].float()
            self.bboxes[:, :2] = self.bboxes[:, :2] - self.bboxes[:, 2:] / 2
            self.bboxes[:, :2] = (self.bboxes[:, :2] + 1) * t_sz / 2
            self.bboxes[:, 2:] = self.bboxes[:, 2:] * t_sz
            self.bboxes = self.bboxes.long()

    def np_to_tensor(self):
        "Convert np.ndarray to Pytorch Compatible Image"
        self.img_display = np.copy(self.img)
        self.img_model = Image(pil2tensor(self.img, np.float32).div_(255))

    def tensor_to_np(self):
        "Convert Tensors to OpenCV Comaptible Types"
        # self.img = image2np(self.img.data * 255).astype(np.uint8)
        # cv2.cvtColor(src=self.img, dst=self.img, code=cv2.COLOR_BGR2RGB)
        # self.img_display = np.copy(self.img)
        self.bboxes = self.bboxes.tolist()
        self.preds = self.preds.tolist()
        self.scores = self.scores.tolist()
        # Convert bboxes to list of top-left and bottom right co-ordinates
        _bboxes = []
        for _bbox in self.bboxes:
            top_left = (_bbox[1], _bbox[0])
            bottom_right = (
                _bbox[1] + int(_bbox[3] / 1.7),
                _bbox[0] + int(_bbox[2] / 1.5),
            )
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

    def start_detection(self, img):
        "Updates and Returns bboxes, preds, scores and classes for next video frame"
        self.img = img
        self.np_to_tensor()
        # Model Prediction
        with torch.no_grad():
            self.model_pred = self.model(self.img_model.data.unsqueeze_(0))
        self.bboxes, self.scores, self.preds = process_output(
            self.model_pred, i=0, detect_thresh=0.55
        )
        self.supress_outputs()
        self.tensor_to_np()
        # self.show()
        return (
            self.bboxes,
            self.preds,
            self.scores,
            self.classes,
        )
