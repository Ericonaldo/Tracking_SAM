import numpy as np
import cv2
from collections import deque
import tracking_SAM.aott
import tracking_SAM.plt_clicker
import tracking_SAM.web_clicker
from segment_anything import sam_model_registry, SamPredictor
import torch

def find_the_next_bbox(boxes):
    """
    Try to find the next object to be grasped, based on the probability of the detection, and the size of the bbox.
        boxes: A list of boxes, each box is an object contains the properties of the bounding box, including
            xyxy - the coordinates of the box as an array [x1,y1,x2,y2]
            cls - the ID of object type
            conf - the confidence level of the model about this object. If it's very low, like < 0.5, then you can just ignore the box.
    """
    # Sort the boxes by the confidence level, and the size of the bbox
    boxes = sorted(boxes, key=lambda x: x.conf, reverse=True)
    if boxes[0].conf < 0.5:
        return None
    # Find the appropriate box first, img size is 960x640
    for box in boxes:
        if box.xyxy[2] - box.xyxy[0] < 550 and box.xyxy[3] - box.xyxy[1] < 400:
            return box
    return boxes[0]

class main_tracker:
    def __init__(self, sam_checkpoint, aot_checkpoint,
                 sam_model_type="vit_h", device="cuda", anno_type="plt_clicker", obj_num=1):
        assert anno_type in ["plt_clicker", "web_clicker", "yolo"]
        self.obj_num = obj_num
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)
        self.anno_type = anno_type
        self.imgs = deque()

        self.vos_tracker = tracking_SAM.aott.aot_segmenter(aot_checkpoint)

        if anno_type == "yolo":
            from ultralytics import YOLO
            # different model size
            # self.det_model = YOLO('yolov8n.pt')
            # self.det_model = YOLO('yolov8s.pt')
            # self.det_model = YOLO('yolov8m.pt')
            # self.det_model = YOLO('yolov8l.pt')
            self.det_model = YOLO('yolov8x.pt')

        self.reset_engine()
    
    def annotate_init_frame(self, img):
        assert self.anno_type in ["plt_clicker", "yolo"]

        if self.anno_type == "plt_clicker":
            anno = tracking_SAM.plt_clicker.Annotator(img, self.sam_predictor)
            anno.main()  # blocking call
            mask_np_hw = anno.get_mask()

            mask_np_hw = mask_np_hw.astype(np.uint8)
            mask_np_hw[mask_np_hw > 0] = 1
        elif self.anno_type == 'yolo':
            results = self.det_model.predict(img)
            bbox = find_the_next_bbox(results.boxes)
            if bbox is None:
                return False
            self.sam_predictor.set_image(img)
            assert len(bbox.xyxy) == 1
            input_box = bbox.xyxy[0].cpu().numpy()

            masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            mask_np_hw = masks[0].astype(np.uint8)

        self.imgs.append([img, mask_np_hw])

        return True
    
    def set_init_frame(self, img, mask_np_hw):
        self.imgs.append([img, mask_np_hw])

    def start_tracking(self):
        if len(self.imgs) == 0:
            return
        img, mask_np_hw = self.imgs.popleft()
        self.vos_tracker.add_reference_frame(img, mask_np_hw)

        self.tracking = True

    def regist_init_frame(self, img=None):
        if img is None:
            if len(self.imgs) > 0:
                img = self.imgs.popleft()[0]
        
        if self.anno_type == "web_clicker":
            assert img is not None, "img should not be None!"
            self.anno = tracking_SAM.web_clicker.Annotator(img, self.sam_predictor)
        elif self.anno_type == "plt_clicker":
            if img is None:
                print("img is none")
                return
            self.anno = tracking_SAM.plt_clicker.Annotator(img, self.sam_predictor)

    def update_click_pos(self, click_pos):
        assert self.anno_type == "web_clicker"

        mask = self.anno.predict(click_pos)
        return mask

    def propagate_one_frame(self, img):
        assert self.tracking, "Please call annotate_init_frame() first!"
        pred_np_hw = self.vos_tracker.propagate_one_frame(img)
        return pred_np_hw
    
    def reset_engine(self):
        self.vos_tracker.reset_engine()
        self.tracking = False
    
    def is_tracking(self):
        return self.tracking
