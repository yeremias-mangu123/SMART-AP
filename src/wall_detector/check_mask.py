import sys
import os
import cv2
import numpy as np
import torch

sys.path.insert(0, '.')
from src.wall_detector.dataset import CubiCasaWallDataset

def debug_dataset():
    ds = CubiCasaWallDataset('data/cubicasa5k/cubicasa5k/high_quality', image_size=512)
    img_tensor, mask_tensor = ds[0]
    mask_np = mask_tensor.squeeze().numpy()
    
    print(f"Mask shape: {mask_np.shape}")
    print(f"Unique values in mask: {np.unique(mask_np)}")
    print(f"Mask sum: {mask_np.sum()}")
    
    # Save with explicit 255 multiplication
    mask_to_save = (mask_np * 255).astype(np.uint8)
    cv2.imwrite('artifacts/debug_dataset/check_mask.png', mask_to_save)
    print("Saved artifacts/debug_dataset/check_mask.png")

if __name__ == '__main__':
    debug_dataset()
