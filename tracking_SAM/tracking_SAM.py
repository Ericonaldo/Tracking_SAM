import numpy as np
import cv2
from PIL import Image
import os
from collections import deque
import tracking_SAM.aott
import tracking_SAM.plt_clicker
import tracking_SAM.web_clicker
from segment_anything import sam_model_registry, SamPredictor
import torch

def is_valid_bbox(bbox, width_cutoff, height_cutoff):
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    if bbox_w < width_cutoff and bbox_h < height_cutoff:
        return True
    else:
        return False

def find_the_next_bbox(boxes_xyxy, logits, phrases):
    # Sort the boxes by the confidence level, and the size of the bbox
    bbox_sorted_idx = np.argsort(logits)
    confidences = logits[bbox_sorted_idx]
    if confidences[0] < 0.1:
        return None
    sorted_bbox = boxes_xyxy[bbox_sorted_idx].cpu().numpy()
    valid_sorted_bbox = [bbox for bbox in sorted_bbox if is_valid_bbox(bbox, 550, 400)]
    # Find the appropriate box first, img size is 960x640
    for box in sorted_bbox:
        if is_valid_bbox(box, 550, 400):
            return box
    return valid_sorted_bbox[0]

class main_tracker:
    def __init__(self, sam_checkpoint, aot_checkpoint, ground_dino_checkpoint,
                 sam_model_type="vit_h", device="cuda", anno_type="plt_clicker", obj_num=1):
        assert anno_type in ["plt_clicker", "web_clicker", "auto"]
        self.obj_num = obj_num
        self.device = device
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)
        self.anno_type = anno_type
        self.imgs = deque()

        self.vos_tracker = tracking_SAM.aott.aot_segmenter(aot_checkpoint)

        if anno_type == "auto":
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
            import groundingdino.datasets.transforms as T
            from groundingdino.util import box_ops
            from groundingdino.util.inference import predict

            cur_dir = os.path.dirname(os.path.abspath(__file__))
            config_file_path = os.path.join(cur_dir,
                                            'third_party',
                                            'GroundingDINO',
                                            'groundingdino',
                                            'config',
                                            'GroundingDINO_SwinT_OGC.py')

            self.det_model = build_model(SLConfig.fromfile(config_file_path))

            checkpoint = torch.load(ground_dino_checkpoint, map_location='cpu')
            self.det_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            self.det_model.eval()
            self.det_model = self.det_model.to(device)

        self.reset_engine()
    
    def annotate_init_frame(self, img):
        assert self.anno_type in ["plt_clicker", "auto"]

        if self.anno_type == "plt_clicker":
            anno = tracking_SAM.plt_clicker.Annotator(img, self.sam_predictor)
            anno.main()  # blocking call
            mask_np_hw = anno.get_mask()

            mask_np_hw = mask_np_hw.astype(np.uint8)
            mask_np_hw[mask_np_hw > 0] = 1
        elif self.anno_type == 'auto':
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),  # not acutally random. It selects from [800].
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            img_chw, _ = transform(Image.fromarray(img), None)
            boxes, logits, phrases = predict(
                model=self.det_model, 
                image=img_chw, 
                caption="box . cup . drill . toyhammer . bucket . bottle . broken can . duck . ball . bowl . string . rope . object", 
                box_threshold= 0.15, 
                text_threshold=0.25,
                device=self.device
            )

            H, W, _ = img.shape

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

            bbox = find_the_next_bbox(boxes_xyxy, logits, phrases)
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
