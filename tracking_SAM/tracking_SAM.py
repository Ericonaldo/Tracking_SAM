import numpy as np
import cv2
from collections import deque
import tracking_SAM.aott
import tracking_SAM.plt_clicker
import tracking_SAM.web_clicker
from segment_anything import sam_model_registry, SamPredictor

class main_tracker:
    def __init__(self, sam_checkpoint, aot_checkpoint,
                 sam_model_type="vit_h", device="cuda", anno_type="plt_clicker", obj_num=1):
        self.obj_num = obj_num
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)
        self.anno_type = anno_type
        self.imgs = deque()

        self.vos_tracker = tracking_SAM.aott.aot_segmenter(aot_checkpoint)

        self.reset_engine()
    
    def annotate_init_frame(self, img):
        assert self.anno_type == "plt_clicker"

        anno = tracking_SAM.plt_clicker.Annotator(img, self.sam_predictor)
        anno.main()  # blocking call
        mask_np_hw = anno.get_mask()

        mask_np_hw = mask_np_hw.astype(np.uint8)
        mask_np_hw[mask_np_hw > 0] = 1  # TODO(roger): support multiple objects?

        self.imgs.put([img, mask_np_hw])

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
