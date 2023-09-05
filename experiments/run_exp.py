import os
import io
import PIL
import json
import torch
import random
import pyrallis
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from config import RunConfig
from pycocotools import mask
from typing import List, Dict, Union
from utils import vis_utils
from diffusers import DDIMScheduler, DDIMInverseScheduler
from pipeline_scribble_guide import ScribbleGuidePipeline, AttentionStore
from transformers import BlipForConditionalGeneration, BlipProcessor

NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    
    model = ScribbleGuidePipeline.from_pretrained(
        stable_diffusion_version
    ).to(device)
    
    return model


def run(model,
        prompt,
        token_masks,
        attention_store,
        attention_resolution,
        latents,
        generator,
        config: RunConfig
        ):
    results = model(
        prompt=prompt,
        token_masks=token_masks,
        attention_store=attention_store,
        attention_resolution=attention_resolution,
        latents=latents,
        generator=generator,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        run_standard=config.run_standard,
        scale_factor=config.scale_factor,
        scale_range=config.scale_range,
    )
    
    image = results.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    captioner_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(captioner_id)
    caption_generator = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float, low_cpu_mem_usage=True)
    
    model = load_model(config)
    
    tokenizer = model.tokenizer
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    model.inverse_scheduler = DDIMInverseScheduler.from_config(model.scheduler.config)
    
    
    
    
