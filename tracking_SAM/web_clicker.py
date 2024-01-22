
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Forbidden  Key: QSFKL


class Annotator(object):
    def __init__(self, img_np, sam_predictor, save_path=None):
        self.sam_predictor = sam_predictor
        self.save_path = save_path
        self.img = img_np.copy()
        self.sam_predictor.set_image(self.img)
        self.reset()

    def reset(self):
        self.clicks = np.empty([0, 2], dtype=np.int64)
        self.pred = np.zeros(self.img.shape[:2], dtype=np.uint8)

    def predict(self, clicks):
        x, y = clicks
        self.clicks = np.append(self.clicks, np.array(
                [[x, y]], dtype=np.int64), axis=0)
        input_label = np.ones((self.clicks.shape[0], ))
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=self.clicks,
            point_labels=input_label,
            multimask_output=False,
        )
        self.pred = masks[0].astype(np.uint8)
        self.tracking = True
        return self.pred

if __name__ == "__main__":
    from segment_anything import sam_model_registry, SamPredictor
    sam_checkpoint = "../pretrained_weights/sam_vit_h_4b8939.pth"  # default model

    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    img_path = "../sample_data/DAVIS_bear/images/00000.jpg"
    img_np = np.array(Image.open(img_path))
    anno = Annotator(img_np, predictor, save_path="/tmp/00000.png")
    anno.main()

    print("Done!")
    print(anno.get_mask().shape)
