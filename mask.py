import sys

sys.path.append("./AdelaiDet/")

import cv2
import torch
from torchvision.transforms import ToPILImage

from pathlib import Path
from argparse import Namespace
from PIL import Image

from typing import List

from detectron2.engine.defaults import DefaultPredictor
from AdelaiDet.adet.config import get_cfg

# dealing with upsample warnings: https://github.com/pytorch/pytorch/issues/50209
torch.warnings.filterwarnings(action="ignore")


class Masking:
    def __init__(self, model_path="./model_final.pth"):
        default_args = {
            "config_file": "./masking/SOLOv2.yaml",
            "webcam": False,
            "video_input": None,
            "input": [],
            "output": None,
            "confidence_threshold": 0.3,
            "opts": [
                "MODEL.WEIGHTS",
                model_path,
            ],
        }
        cfg = self.setup_cfg(Namespace(**default_args))
        self.predictor = DefaultPredictor(cfg)

        self.toimg = ToPILImage()

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
        cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            args.confidence_threshold
        )
        cfg.freeze()
        return cfg

    def run_predict(self, image_path: str):
        if not Path(image_path).exists():
            print(f"Image {image_path} does not exists")
            exit(0)

        image = cv2.imread(image_path)
        return self.predictor(image)

    def predict(self, image: Image.Image):
        return self.predictor(image)

    def predict_masks(self, images: List[Image.Image]):
        generated_masks = []

        for image in images:
            generated_masks.append(self.predict(image))

        out_imgs = []
        for mask_struct in generated_masks:
            # This is for a single image, merge all of the instances
            res_img = None
            for image in mask_struct["instances"].pred_masks:
                if res_img is None:
                    res_img = torch.zeros_like(image, dtype=torch.uint8)
                res_img[image == False] = 0
                res_img[image == True] = 255

            out_imgs.append(self.toimg(res_img).convert("RGB"))

        return out_imgs
