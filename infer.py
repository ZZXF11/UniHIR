import os
import re
import time
import json
import torch
import random
import requests
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
from utils import UniHIRInferencer, load_model, invert_image, save_results

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):

    model, tokenizer, vae_model, vae_transform, vae_transform_gen, vit_transform, new_token_ids = load_model(args.model_path)

    inferencer = UniHIRInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform,
        vae_transform_gen=vae_transform_gen,
        new_token_ids=new_token_ids
    )

    inference_hyper = dict(
        max_think_token_n=10000,
        do_sample=False,
        cfg_text_scale=1.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )

    original_image = Image.open(args.img_path)
    input_image = invert_image(original_image)
    original_size = original_image.size

    prompt = '请对图中破损位置进行定位，并输出破损定位图。'
    output_dict = inferencer(image=input_image, 
                            text=prompt,
                            und_only = False,
                            gen_only = False,
                            reflection_count = args.ref_count,
                            think=False, **inference_hyper)

    save_results(output_dict, args.save_path, original_size, args.img_path)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='UniHIR Inference Script')
    parser.add_argument(
        '--img_path',
        type=str,
        default='examples/FS_12_159_2.jpg',
        help='Path to input damaged historical document image'
    )
    parser.add_argument(
        '--ref_count',
        type=int,
        default=6,
        help='Number of refinement iterations (default: 6)'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./results',
        help='Directory to save inference results'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./UniHIR',
        help='Path to the pretrained UniHIR model directory'
    )
    args = parser.parse_args()
    main(args)