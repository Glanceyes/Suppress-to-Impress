import os
import cv2
import json
import spacy
import torch
import pylab
import random
import argparse
import matplotlib
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from json import encoder
from pycocotools.coco import COCO
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

from typing import List, Dict
from pipeline_attend_and_excite import AttendAndExcitePipeline
from config import RunConfig
from run import run_on_prompt, get_indices_to_alter
from utils import vis_utils
from utils.ptp_utils import AttentionStore, aggregate_attention


NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


class COCODataset(Dataset) :
    def __init__(self, root_dir, image_set, transform=None, viz=False):
        super(COCODataset, self).__init__()

        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.viz = viz
        self.captions = {}
        
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.image_set + '.json'))
        whole_image_ids = self.coco.getImgIds()
        
        self.image_ids = []
        self.no_anno_list = []
        self.image_paths = []
        
        image_num = 0
        for idx in whole_image_ids:
            annotations_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
            if len(annotations_ids) == 0 : 
                self.no_anno_list.append(idx)
            else: 
                self.image_ids.append(idx)
                self.image_paths.append(os.path.join(self.root_dir, self.image_set, self.coco.loadImgs(idx)[0]['file_name']))
                image_num += 1

        self.load_classes()
        self.load_captions()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annot = self.load_annotations(idx)
        cap = self.load_caption(idx)
        image_path = self.image_paths[idx]

        if self.viz :
            self.showAnns(image, annot)
            plt.show()

        return image, annot, cap, image_path
    
    def load_caption(self, idx):
        return self.image_captions[self.image_ids[idx]]
    
    def load_captions(self):
        self.image_caption_path = os.path.join(self.root_dir, 'annotations', 'captions_' + self.image_set + '.json')
        image_captions = json.load(open(self.image_caption_path, 'r'))['annotations']
        
        self.image_captions = {}
        
        tqdm.write('Loading captions...')
        
        for image_caption in tqdm(image_captions):
            self.image_captions[image_caption['image_id']] = image_caption['caption'] 
            
            
    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        #image_path = os.path.join(self.root_dir, self.image_set, image_info['file_name'])
        image_path = self.image_paths[image_index]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.
    
    def load_image_id(self, image_index):
        return self.image_ids[image_index]
    
    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.viz_classes, self.classes = {}, {}
        for c in categories:
            self.viz_classes[c['name']] = c['id']
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.viz_classes.items():
            self.labels[value] = key
            
    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 6))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            if a['category_id'] in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
                continue

            annotation = np.zeros((1, 6))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] =  self.classes[self.labels[a['category_id']]]
            annotation[0, 5] = a['category_id']

            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations
            
    def showAnns(self, image, anns):
        ax = plt.gca()
        ax.imshow(image)
        ax.set_autoscale_on(False)
        polygons, colors = [], []

        for ann in anns:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, _, category_id = ann
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = [[bbox_x1, bbox_y1], [bbox_x1, bbox_y2], [bbox_x2, bbox_y2], [bbox_x2, bbox_y1]]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
            colors.append(c)
            ax.text(bbox_x1, bbox_y1, self.labels[category_id], color=c)

        p = PatchCollection(polygons, facecolor='none', edgecolors=colors, linewidths=2)
        ax.add_collection(p)
        plt.axis('off'); plt.xticks([]); plt.yticks([])



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
        mask, attention_store, tokens, output_dir, res=16, 
        from_where=("up", "down", "mid"), is_cross=True, select=0
    ):
        attention_maps = aggregate_attention(
            attention_store, res=res, 
            from_where=from_where, is_cross=is_cross, select=select
        )
        
        output_attn_dir = f"{output_dir}/attention_maps"
        
        os.makedirs(output_attn_dir, exist_ok=True)

        for token, i in tokens.items():
            image = attention_maps[:, :, i]
            
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
        img, output_dir, tokens
    ):
        
        input_img = img.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(input_img)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)
        
        with torch.no_grad():
            output = seg_model(input_batch)['out'][0]
        output_prediction = output.argmax(0).detach().cpu()
    
        result_img = Image.fromarray(output_prediction.byte().cpu().numpy()).resize(input_img.size)
    
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        
        result_img.putpalette(colors)

        # save image
        result_img.save(f"{output_dir}/seg_sem.png")
        
        output_mask_dir = f"{output_dir}/masks"
        os.makedirs(output_mask_dir, exist_ok=True)
        
        for token in tokens:
            if token not in classnames_list:
                continue
            mask = torch.zeros_like(output_prediction)
            mask = torch.where(output_prediction == class_voc_to_idx[token], 1, 0)
            
            with open(f"{output_mask_dir}/{token}.npy", "wb") as f:
                np.save(f, mask.numpy())
                
            with open(f"{output_mask_dir}/{token}.png", "wb") as f:
                Image.fromarray(mask.byte().cpu().numpy() * 255).save(f)
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="../../datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/runs_multi"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21
    )
    parser.add_argument(
        "--use_float16",
        type=int,
        default=1
    )
    
    opt = parser.parse_args()
    
    dataset_dir = opt.dataset_dir
    output_dir = opt.output_dir
    seed = opt.seed
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    use_float_16 = opt.use_float16
    
    dataset = COCODataset(root_dir=dataset_dir, image_set='train2017', viz=False)
    dataloader = DataLoader(dataset, shuffle=False)
    
    seg_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).to(device)
    seg_model.eval()
    
    nlp = spacy.load('en_core_web_sm')
    

    with open(os.path.join(opt.dataset_dir, "annotations", "classnames_voc_filtered.json"), "r") as f:
        classnames_list = json.load(f).keys()
        
    
    class_voc_to_idx = {}

    with open(os.path.join(opt.dataset_dir, "annotations", "classnames_voc.txt"), 'r') as f:
        for idx, line in enumerate(f.readlines()):
            class_voc_to_idx[line.strip()] = idx
            
    idx_to_class_voc = {}

    for item in class_voc_to_idx.items():
        idx_to_class_voc[item[1]] = item[0]


    stable = AttendAndExcitePipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16 if use_float_16 else None).to(device)
    
    with torch.no_grad():
        for idx, (img, annot, cap, img_path) in enumerate(dataloader):
           
            prompt = cap[0].lower()
            doc = nlp(prompt)

            tokens = defaultdict(list)
                    
            for token in doc:
                if token.text not in tokens.keys() and token.text in classnames_list:
                    tokens[token.text] = token.i + 1
                    
            if tokens == {}:
                print(f"Skipping {idx} since no tokens found.")
                continue
            
            token_indices = [tokens[token] for token in tokens]
            
            img = img[0].detach().cpu().numpy()
            img = np.array(img * 255, dtype=np.uint8)
            
            img_id = dataset.load_image_id(idx)
            
            output_img_dir = f"{output_dir}/{img_id}"
            os.makedirs(output_img_dir, exist_ok=True)
            
            # save image as cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{output_img_dir}/image.png", img)
            
            # save cap as txt
            with open(f"{output_img_dir}/cap.txt", 'w') as f:
                f.write(prompt)
    
            g = torch.Generator('cuda').manual_seed(seed)
            prompts = [prompt]
            controller_sd = AttentionStore()
            
            image_sd = run_and_display(prompts=prompts,
                                    controller=controller_sd,
                                    indices_to_alter=token_indices,
                                    generator=g,
                                    run_standard_sd=True,
                                    display_output=False)
            
            output_sd_dir = f"{output_img_dir}/standard_sd"
            os.makedirs(output_sd_dir, exist_ok=True)
            
            image_sd_np = np.array(image_sd)
            image_sd_cv = cv2.cvtColor(image_sd_np, cv2.COLOR_RGB2BGR)
            
            # save image
            cv2.imwrite(f"{output_sd_dir}/result.png", image_sd_cv)
                    
            save_attention(
                image_sd,
                controller_sd, 
                tokens=tokens,
                res=16, 
                from_where=("up", "down", "mid"), 
                output_dir=output_sd_dir
            )
            
            save_masks(
                image_sd,
                output_dir=output_sd_dir,
                tokens=tokens
            )
            
            controller_ae = AttentionStore()
            image_ae = run_and_display(prompts=prompts,
                                controller=controller_ae,
                                indices_to_alter=token_indices,
                                generator=g,
                                run_standard_sd=False,
                                display_output=False)
            
            
            output_attend_excite_dir = f"{output_img_dir}/attend_excite"
            os.makedirs(output_attend_excite_dir, exist_ok=True)
            
            image_ae_np = np.array(image_ae)
            image_ae_cv = cv2.cvtColor(image_ae_np, cv2.COLOR_RGB2BGR)
            
            # save image
            cv2.imwrite(f"{output_attend_excite_dir}/result.png", image_ae_cv)
            
            save_attention(
                image_ae,
                controller_ae,
                tokens=tokens,
                res=16, 
                from_where=("up", "down", "mid"), 
                output_dir=output_attend_excite_dir,
            )
            
            save_masks(
                image_ae,
                output_dir=output_attend_excite_dir,
                tokens=tokens
            )

    return



if __name__ == '__main__':
    main()