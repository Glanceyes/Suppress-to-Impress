import os
import cv2
import json
import torch
import pylab
import random
import spacy
import argparse
import matplotlib
import detectron2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from json import encoder
from pycocotools.coco import COCO
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from typing import List, Dict
from pipeline_attend_and_excite import AttendAndExcitePipeline
from config import RunConfig
from run import run_on_prompt, get_indices_to_alter
from utils import vis_utils
from utils.ptp_utils import AttentionStore, aggregate_attention

NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

def main():
    
    # configurable parameters (see RunConfig for all parameters)
    # scale factor- intensity of shift by gradient
    # thresholds- a dictionary for iterative refinement mapping the iteration number to the attention threshold
    # max_iter_to_alter- maximal inference timestep to apply Attend-and-Excite
    def run_and_display(prompts: List[str],
                        controller: AttentionStore,
                        indices_to_alter: List[int],
                        generator: torch.Generator,
                        run_standard_sd: bool = False,
                        scale_factor: int = 20,
                        thresholds: Dict[int, float] = {0: 0.05, 10: 0.5, 20: 0.8},
                        max_iter_to_alter: int = 25,
                        display_output: bool = False):
        config = RunConfig(prompt=prompts[0],
                        run_standard_sd=run_standard_sd,
                        scale_factor=scale_factor,
                        thresholds=thresholds,
                        max_iter_to_alter=max_iter_to_alter)
        image = run_on_prompt(model=stable,
                            prompt=prompts,
                            controller=controller,
                            token_indices=indices_to_alter,
                            seed=generator,
                            config=config)
        return image
    
    def save_attention(
        mask, attention_store, token, token_id, output_dir, res=16, 
        from_where=("up", "down", "mid"), is_cross=True, select=0
    ):
        attention_maps = aggregate_attention(
            attention_store, res=res, 
            from_where=from_where, is_cross=is_cross, select=select
        )
        
        output_attn_dir = f"{output_dir}/attention_maps"
        os.makedirs(output_attn_dir, exist_ok=True)

        image = attention_maps[:, :, token_id]
        
        attn_map = np.array(image.detach().cpu())
        # save image npy
        np.save(f"{output_attn_dir}/{token}.npy", attn_map)
        
        image = vis_utils.show_image_relevance(image, mask)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # save as image
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
        
        cv2.imwrite(f"{output_attn_dir}/{token}.png", image)
            
            
    def save_masks(
        img, output_dir, token
    ):
        output = predictor(img)
        masks = output["instances"].pred_masks
        scores = output["instances"].scores.cpu().numpy()
        pred_classes = output["instances"].pred_classes.cpu().numpy()
        
        v = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        
        v = v.draw_instance_predictions(output["instances"].to("cpu"))
        result_image = v.get_image()[:, :, ::-1]
        
        output_mask_dir = f"{output_dir}/masks"
        os.makedirs(output_mask_dir, exist_ok=True)

        # save image
        cv2.imwrite(f"{output_dir}/ins_seg.png", result_image)
        
        masks_to_save = defaultdict(list)
        masks_to_save_high_conf = defaultdict(list)
        
        for i, mask in enumerate(masks):
            score = scores[i]
            cat_id = pred_classes[i]
            if idx_to_cat[cat_id] == token:
                if score > 0.9:
                    mask_high_conf = mask.cpu().numpy()
                    masks_to_save_high_conf[idx_to_cat[cat_id]].append(mask_high_conf)
                if score > 0.5:    
                    mask = mask.cpu().numpy()
                    masks_to_save[idx_to_cat[cat_id]].append(mask)
        
        for cat in masks_to_save_high_conf.keys():
            masks_to_save_high_conf[cat] = np.array(masks_to_save_high_conf[cat])
            masks_to_save_high_conf[cat] = np.sum(masks_to_save_high_conf[cat], axis=0)
            with open(f"{output_mask_dir}/{cat}_well_seg.npy", 'wb') as f:
                np.save(f, masks_to_save_high_conf[cat])
            cv2.imwrite(f"{output_mask_dir}/{cat}_well_seg.png", masks_to_save_high_conf[cat] * 255)
        
        for cat in masks_to_save.keys():
            masks_to_save[cat] = np.array(masks_to_save[cat])
            masks_to_save[cat] = np.sum(masks_to_save[cat], axis=0)
            with open(f"{output_mask_dir}/{cat}.npy", 'wb') as f:
                np.save(f, masks_to_save[cat])
            cv2.imwrite(f"{output_mask_dir}/{cat}.png", masks_to_save[cat] * 255)
        
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="../../datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/runs_single"
    )
    parser.add_argument(
        "--seeds",
        default=[21, 41, 61, 81, 101],
        nargs='+'
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=10
    )
    parser.add_argument(
        "--use_float16",
        type=int,
        default=0
    )
    
    opt = parser.parse_args()
    
    # Detectron2
    setup_logger()
    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    predictor = DefaultPredictor(cfg)
    
    cat_to_idx = {}
    
    classname_path = os.path.join(opt.dataset_dir, "annotations", "classnames_val2017.txt")

    with open(classname_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            cat_to_idx[line.strip().replace(' ', '-')] = idx
            
    idx_to_cat = {}

    for item in cat_to_idx.items():
        idx_to_cat[item[1]] = item[0].replace(' ', '-')
        
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    use_float_16 = opt.use_float16
    stable = AttendAndExcitePipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16 if use_float_16 else None).to(device)
    
    seeds = opt.seeds
    assert len(seeds) > 0 and type(opt.seeds) is list, "Please provide at least one seed"
    
    output_dir = opt.output_dir
    num_iter = opt.num_iter
    
    output_dir = opt.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # allow_cat = [
    #     'carrot', 'mouse', 'orange', 'toaster', 
    # ]
    
    with torch.no_grad():
        tqdm.write("Generating images")
        
        for cat_id, cat in tqdm(idx_to_cat.items()):
            
            # if cat not in allow_cat:
            #     continue
            
            token = cat
            prompt = f"{'a' if token[0] not in ('a', 'e', 'i', 'o', 'u') else 'an'} {token}"
            print(prompt)
            prompts = [prompt]

            output_cat_dir = f"{output_dir}/{cat}"
            os.makedirs(output_cat_dir, exist_ok=True)

            # find index of token in prompt
            token_id = prompt.split().index(token) + 1
            if '-' not in token:
                token_indices = [token_id]
            else:
                token_indices = [token_id, token_id + 2]
    
            for seed in tqdm(seeds):
                tqdm.write(f"Generating images for seed {seed}")
                
                g = torch.Generator('cuda').manual_seed(seed)
                output_cat_seed_dir = f"{output_cat_dir}/{seed}"
                os.makedirs(output_cat_seed_dir, exist_ok=True)
                
                for idx in range(num_iter):
                    controller_sd = AttentionStore()
                    
                    output_iter_dir = f"{output_cat_seed_dir}/{idx}"
                    os.makedirs(output_iter_dir, exist_ok=True)
                    
                    image_sd = run_and_display(prompts=prompts,
                                            controller=controller_sd,
                                            indices_to_alter=token_indices,
                                            generator=g,
                                            run_standard_sd=True,
                                            display_output=False)
                    
                    output_sd_dir = f"{output_iter_dir}/standard_sd"
                    os.makedirs(output_sd_dir, exist_ok=True)
                    
                    image_sd_np = np.array(image_sd)
                    image_sd_cv = cv2.cvtColor(image_sd_np, cv2.COLOR_RGB2BGR)
                    
                    # save image
                    cv2.imwrite(f"{output_sd_dir}/result.png", image_sd_cv)
                    
                    save_attention(
                        image_sd,
                        controller_sd, 
                        token=token,
                        token_id=token_id,
                        res=16, 
                        from_where=("up", "down", "mid"), 
                        output_dir=output_sd_dir,
                    )
                    
                    save_masks(
                        np.array(image_sd),
                        output_dir=output_sd_dir,
                        token=token
                    )
                    
                    controller_ae = AttentionStore()
                    image_ae = run_and_display(prompts=prompts,
                                        controller=controller_ae,
                                        indices_to_alter=token_indices,
                                        generator=g,
                                        run_standard_sd=False,
                                        display_output=False)
                    
                    output_attend_excite_dir = f"{output_iter_dir}/attend_excite"
                    os.makedirs(output_attend_excite_dir, exist_ok=True)
                    
                    image_ae_np = np.array(image_ae)
                    image_ae_cv = cv2.cvtColor(image_ae_np, cv2.COLOR_RGB2BGR)
                    
                    # save image
                    cv2.imwrite(f"{output_attend_excite_dir}/result.png", image_ae_cv)
                    
                    save_attention(
                        image_ae,
                        controller_ae,
                        token=token,
                        token_id=token_id,
                        res=16, 
                        from_where=("up", "down", "mid"), 
                        output_dir=output_attend_excite_dir
                    )
                    
                    save_masks(
                        np.array(image_ae),
                        output_dir=output_attend_excite_dir,
                        token=token
                    )

    return



if __name__ == '__main__':
    main()