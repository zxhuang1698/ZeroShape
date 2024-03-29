# These code is adapted from https://github.com/dreamgaussian/dreamgaussian/blob/main/process.py
# The original code is licensed under the MIT License.


import os
import glob
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import rembg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--model', default='u2net', type=str, help="rembg model, see https://github.com/danielgatis/rembg#models")
    parser.add_argument('--size', default=224, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=bool, default=True, help="recenter, potentially not helpful for multiview zero123")    
    opt = parser.parse_args()

    session = rembg.new_session(model_name=opt.model)

    if os.path.isdir(opt.path):
        print(f'[INFO] processing directory {opt.path}...')
        files = glob.glob(f'{opt.path}/*')
    else: # isfile
        files = [opt.path]

    out_dir = "./my_examples"
    out_images_dir = os.path.join(out_dir, 'images')
    out_masks_dir = os.path.join(out_dir, 'masks')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)
    
    for file in files:

        out_base = os.path.basename(file).split('.')[0]
        out_rgba = os.path.join(out_images_dir, out_base + '.png')
        out_mask = os.path.join(out_masks_dir, out_base + '.png')

        # load image
        print(f'[INFO] loading image {file}...')
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        
        # carve background
        print(f'[INFO] background removal...')
        carved_image = rembg.remove(image, session=session) # [H, W, 4]
        mask = carved_image[..., -1] > 0

        # recenter
        if opt.recenter:
            print(f'[INFO] recenter...')
            final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
            
            coords = np.nonzero(mask)
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            h = x_max - x_min
            w = y_max - y_min
            desired_size = int(opt.size * (1 - opt.border_ratio))
            scale = desired_size / max(h, w)
            h2 = int(h * scale)
            w2 = int(w * scale)
            x2_min = (opt.size - h2) // 2
            x2_max = x2_min + h2
            y2_min = (opt.size - w2) // 2
            y2_max = y2_min + w2
            final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
            
        else:
            final_rgba = carved_image

        final_mask = (final_rgba[..., -1] > 0).astype(np.uint8) * 255
        
        # write image
        cv2.imwrite(out_rgba, final_rgba)
        cv2.imwrite(out_mask, final_mask)