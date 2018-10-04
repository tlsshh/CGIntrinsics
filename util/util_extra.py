import os
import numpy as np
from PIL import Image
import torch
import util


def save_shading_image(image, path, filename):
    # Transform to PILImage
    image_np = np.squeeze(image.cpu().float().numpy())
    # image_np = (image_np + 1.0) / 2.0 * 255.0
    image_np[image_np<1e-6] = 1e-6
    image_np[image_np>1] = 1
    # image_norm = (image_np - np.min(image_np))/np.ptp(image_np) * 255# Normalized [10,210]
    image_norm = image_np*255.0
    image_norm = image_norm.astype(np.uint8)
    image_pil = Image.fromarray(image_norm, mode='L')
    # Save Image
    util.mkdir(path)
    image_pil.save(os.path.join(path,filename))


def save_reflect_image(image, path, filename):
    # Transform to PILImage
    image_np = np.transpose(image.cpu().float().numpy(), (1, 2, 0))
    image_np[image_np<1e-6] = 1e-6
    image_np[image_np>1] = 1
    image_norm = image_np*255.0
    # image_norm = (image_np + 1.0) / 2.0 * 255.0
    # image_norm = (image_np - np.min(image_np))/np.ptp(image_np) * 200 + 10 # Normalized [10,210]
    image_norm = image_norm.astype(np.uint8)
    image_pil = Image.fromarray(image_norm, mode='RGB')
    # Save Image
    util.mkdir(path)
    image_pil.save(os.path.join(path,filename))


def save_rgb_image(image, path, filename):
    # Transform to PILImage
    image_np = np.transpose(image.cpu().float().numpy(), (1, 2, 0)) * 255.0
    image_np = image_np.astype(np.uint8)
    image_pil = Image.fromarray(image_np, mode='RGB')
    # Save Image
    util.mkdir(path)
    image_pil.save(os.path.join(path,filename))

def visual_adjust(R, S):
    max_R = np.percentile(R, 95)
    factor = max_R * 1.5
    R = R / factor
    S = S * factor
    return R, S

def visualize_results(path, pred_R, pred_S, srgb_img, chromaticity):
    # 1 channel
    pred_R = torch.exp(pred_R).repeat(3, 1, 1)
    pred_R = torch.mul(chromaticity, pred_R)
    pred_S = torch.exp(pred_S)
    pred_R, pred_S = visual_adjust(pred_R, pred_S)
    save_reflect_image(pred_R, path, 'R.png')
    save_shading_image(pred_S, path, 'S.png')
    save_rgb_image(srgb_img, path, 'rgb.png')
    pass



