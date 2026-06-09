"""
Dataset loader for CubiCasa5K floor plan dataset.
Converts SVG wall annotations to binary segmentation masks for training.
Uses svgelements for robust SVG parsing including transforms.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from svgelements import SVG, Path, Polygon, Polyline, Rect, Matrix, Group


def parse_svg_walls(svg_path, target_size=(256, 256)):
    """Parse wall elements from CubiCasa5K SVG annotation file using svgelements.
    
    Handles transforms, paths, and various shape types accurately.
    """
    if not os.path.exists(svg_path):
        return None
    
    try:
        svg = SVG.parse(svg_path)
    except Exception as e:
        print(f"Error parsing SVG {svg_path}: {e}")
        return None
    
    # Get SVG dimensions from viewBox or width/height
    viewbox = svg.viewbox
    if viewbox:
        svg_w, svg_h = viewbox.width, viewbox.height
    else:
        svg_w, svg_h = svg.width, svg.height
    
    # Fallback if dimensions are missing
    if svg_w == 0 or svg_h == 0:
        svg_w, svg_h = 256, 256

    # Create blank mask at SVG resolution
    mask = np.zeros((int(svg_h), int(svg_w)), dtype=np.uint8)
    
    # Calculate adaptive wall thickness based on SVG resolution
    # Walls in floor plan images are typically 3-8 pixels thick at 512px
    # At SVG resolution (often 1000+px), walls need to be thicker
    svg_scale = max(int(svg_w), int(svg_h)) / 512.0
    wall_thickness = max(3, int(6 * svg_scale))  # Scale thickness with resolution
    
    def process_element(elem, is_wall_parent=False):
        # Get element properties
        values = getattr(elem, 'values', {})
        elem_class = str(values.get('class', '')).lower()
        elem_id = str(getattr(elem, 'id', '')).lower()
        
        # Check for wall markers (whitelist)
        has_wall_marker = any(wc in elem_class or wc in elem_id for wc in ['wall', 'railing'])
        
        # Exclude common false positives (blacklist)
        is_false_positive = any(ex in elem_class or ex in elem_id for ex in ['cabinet', 'wallpaper', 'decor', 'furniture'])
        
        # Exclude explicit non-walls
        is_explicit_non_wall = any(ex in elem_class or ex in elem_id for ex in ['door', 'window', 'threshold', 'equipment', 'appliance', 'sink', 'toilet', 'shower'])
        
        # Determine if this element is a wall
        is_this_wall = (is_wall_parent or has_wall_marker) and not (is_false_positive or is_explicit_non_wall)
        
        # If it's a wall and has geometry, draw it
        tname = elem.__class__.__name__
        if is_this_wall and tname in ('Path', 'Polygon', 'Polyline', 'Rect'):
            try:
                if tname == 'Path':
                    pts = []
                    for i in range(101):
                        p = elem.point(i / 100)
                        pts.append((p.x, p.y))
                else:
                    pts = [(p.x, p.y) for p in elem.points]
                
                if len(pts) >= 2:
                    pts_np = np.array(pts, dtype=np.int32)
                    if len(pts_np) >= 3:
                        # Fill the polygon (covers the wall area)
                        cv2.fillPoly(mask, [pts_np], 1)
                        # Also draw thick outline to ensure thin walls are visible
                        cv2.polylines(mask, [pts_np], isClosed=True, color=1, thickness=wall_thickness)
                    else:
                        cv2.line(mask, tuple(pts_np[0]), tuple(pts_np[1]), 1, thickness=wall_thickness)
            except Exception:
                pass
        
        # Recurse into children if it's a container
        if isinstance(elem, (SVG, Group)):
            pass_wallness = is_this_wall if not is_explicit_non_wall else False
            for sub_elem in elem:
                process_element(sub_elem, pass_wallness)

    process_element(svg)
    
    # Resize to target size
    if mask.shape != target_size:
        mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Dilate mask to match visual wall thickness in the PNG images
    # Walls in rendered floor plans are typically 4-8px thick at 512px
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    return mask


def get_train_transforms(image_size=256):
    """Get augmentation pipeline for training."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.ColorJitter(p=1),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=256):
    """Get transform pipeline for validation/inference."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class CubiCasaWallDataset(Dataset):
    """Dataset for CubiCasa5K wall segmentation using svgelements."""
    
    def __init__(self, root_dir, sample_ids=None, transform=None, image_size=256):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform # Don't set a default here, let caller decide
        self.samples = []
        
        # Discovery logic
        all_subdirs = []
        if sample_ids:
            all_subdirs = [os.path.join(root_dir, s) if not os.path.isabs(s) else s for s in sample_ids]
        else:
            # Recursive search for directories containing model.svg
            for root, dirs, files in os.walk(root_dir):
                if 'model.svg' in files:
                    all_subdirs.append(root)
        
        for sample_dir in all_subdirs:
            img_path = None
            svg_path = os.path.join(sample_dir, 'model.svg')
            
            # Try to find the best image
            for f in os.listdir(sample_dir):
                if f.lower().endswith('.png'):
                    if 'original' in f.lower():
                        img_path = os.path.join(sample_dir, f)
                        break
                    elif img_path is None:
                        img_path = os.path.join(sample_dir, f)
            
            if img_path and os.path.exists(svg_path):
                self.samples.append((img_path, svg_path))
        
        print(f"CubiCasaWallDataset: Found {len(self.samples)} valid samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, svg_path = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = parse_svg_walls(svg_path, target_size=(image.shape[0], image.shape[1]))
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            else:
                mask = mask.astype(np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask


class SimpleWallDataset(Dataset):
    """Fallback dataset using adaptive thresholding."""
    
    def __init__(self, image_paths, transform=None, image_size=256):
        self.image_paths = image_paths
        self.image_size = image_size
        self.transform = transform or get_val_transforms(image_size)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Use adaptive thresholding for better results
            mask = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 15, 8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = mask.astype(np.float32)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask
