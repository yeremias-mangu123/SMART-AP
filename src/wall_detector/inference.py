"""
Inference module for Wall Detection.
Uses a hybrid approach: AI model segmentation + classical CV line detection,
with aggressive deduplication to avoid overlapping wall segments.
"""

import os
import cv2
import numpy as np
import torch
import base64
from typing import List, Dict, Tuple, Optional

from src.wall_detector.model import create_model


class WallDetector:
    """Wall detection inference engine using trained UNet model."""
    
    def __init__(self, weights_path: str = None, device: str = None, image_size: int = 512):
        """
        Args:
            weights_path: Path to trained model weights (.pth file)
            device: 'cuda' or 'cpu' (auto-detected if None)
            image_size: Model input size (can be larger than training size)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.image_size = image_size
        
        # Default weights path
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(__file__), 'weights', 'wall_detector_best.pth'
            )
        
        # Load model metadata first to get correct architecture
        encoder_name = "resnet18" # Default
        self.has_model = False
        checkpoint = None
        
        if os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                encoder_name = checkpoint.get('encoder_name', "resnet18")
                self.image_size = checkpoint.get('image_size', image_size)
                print(f"Detected model architecture: {encoder_name}")
            except Exception as e:
                print(f"Error reading checkpoint metadata: {e}")
        
        # Initialize model with detected architecture
        self.model = create_model(encoder_name=encoder_name)
        
        if checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.has_model = True
                print(f"Loaded wall detector weights from {weights_path}")
                print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
                print(f"  Val IoU: {checkpoint.get('val_iou', 'N/A')}")
            except Exception as e:
                print(f"Error loading model state dict: {e}")
        else:
            print(f"WARNING: No weights found at {weights_path}")
            print("  Will use classical CV only")
        
        self.model.to(self.device)
        self.model.eval()
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        resized = cv2.resize(image, (self.image_size, self.image_size))
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor
    
    def predict_mask(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict wall segmentation mask using AI model."""
        original_h, original_w = image.shape[:2]
        input_tensor = self.preprocess(image).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
        
        mask = (prob > threshold).astype(np.uint8)
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def detect_lines_classical(self, image_bgr: np.ndarray, pixels_per_meter: float, 
                                min_wall_length_m: float = 0.5) -> List[Tuple]:
        """Detect wall lines using classical Computer Vision (Canny + Hough).
        
        This is the primary detection method. It works well on clean floor plans.
        Returns raw line segments as list of (x1, y1, x2, y2) in pixel coordinates.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle varying contrast
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 8
        )
        
        # Clean: remove small noise but keep wall structures
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Dilate slightly to connect broken lines
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel_connect, iterations=1)
        
        # Edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Hough Line Transform
        min_line_px = int(min_wall_length_m * pixels_per_meter)
        max_gap = int(0.5 * pixels_per_meter)  # 50cm gap allowed
        
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=30,
            minLineLength=min_line_px,
            maxLineGap=max_gap
        )
        
        segments = []
        if lines is not None:
            segments = [tuple(line[0]) for line in lines]
        segments.extend(self._detect_structural_lines(image_bgr, pixels_per_meter, min_wall_length_m))
        return segments

    def _detect_structural_lines(self, image_bgr: np.ndarray, pixels_per_meter: float,
                                  min_wall_length_m: float = 0.5) -> List[Tuple]:
        """Detect long horizontal/vertical ink runs from the original floor plan.

        The model sometimes misses thin grey walls. This morphology pass is tuned
        for straight architectural lines, so it can recover those missed spans
        without adding curved door swings as walls.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        min_line_px = max(12, int(min_wall_length_m * pixels_per_meter))

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 31, 9
        )
        dark = cv2.inRange(gray, 0, 205)
        binary = cv2.bitwise_or(cv2.bitwise_or(otsu, adaptive), dark)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

        kernel_len = max(9, min(int(min_line_px * 0.65), 55))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

        segments = []
        segments.extend(self._segments_from_axis_mask(horizontal, axis='h', min_line_px=min_line_px))
        segments.extend(self._segments_from_axis_mask(vertical, axis='v', min_line_px=min_line_px))
        return segments

    def _segments_from_axis_mask(self, mask: np.ndarray, axis: str, min_line_px: float) -> List[Tuple]:
        """Convert a morphology mask into line segments."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        segments = []

        for label_id in range(1, num_labels):
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area < min_line_px * 0.5:
                continue

            if axis == 'h':
                if width < min_line_px or width < height * 3:
                    continue
                cy = y + height / 2
                segments.append((float(x), float(cy), float(x + width), float(cy)))
            else:
                if height < min_line_px or height < width * 3:
                    continue
                cx = x + width / 2
                segments.append((float(cx), float(y), float(cx), float(y + height)))

        return segments

    def _detect_long_horizontal_wall_runs(self, image_bgr: np.ndarray, pixels_per_meter: float,
                                          min_wall_length_m: float) -> List[Tuple]:
        """Recover long straight horizontal wall runs missed near door symbols.

        Corridor walls in floor plans are often broken by door swing graphics.
        This pass only adds dark, thin, horizontal runs and skips repeated
        parallel-line clusters such as stairs.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        min_line_px = max(int(min_wall_length_m * pixels_per_meter), int(0.8 * pixels_per_meter), 24)
        dark = cv2.inRange(gray, 0, 135)
        kernel_len = max(21, min(int(min_line_px * 0.9), 55))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
        candidates = []
        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if width < min_line_px or height > 8 or width < height * 8 or area < width * 0.45:
                continue
            candidates.append({
                "x1": float(x),
                "x2": float(x + width),
                "y": float(y + height / 2),
                "width": float(width),
            })

        recovered = []
        for candidate in candidates:
            nearby_parallel = 0
            for other in candidates:
                if other is candidate:
                    continue
                if abs(other["y"] - candidate["y"]) > max(12.0, 0.60 * pixels_per_meter):
                    continue
                overlap = max(0.0, min(candidate["x2"], other["x2"]) - max(candidate["x1"], other["x1"]))
                shorter = max(1.0, min(candidate["width"], other["width"]))
                if overlap / shorter >= 0.55:
                    nearby_parallel += 1

            # Stairs and hatch marks produce many close, parallel horizontal runs.
            if nearby_parallel >= 2:
                continue

            recovered.append((candidate["x1"], candidate["y"], candidate["x2"], candidate["y"]))

        return recovered

    def _detect_long_vertical_wall_runs(self, image_bgr: np.ndarray, pixels_per_meter: float,
                                        min_wall_length_m: float) -> List[Tuple]:
        """Recover long straight vertical wall runs missed by segmentation."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        min_line_px = max(int(min_wall_length_m * pixels_per_meter), int(0.8 * pixels_per_meter), 24)
        dark = cv2.inRange(gray, 0, 145)
        kernel_len = max(21, min(int(min_line_px * 0.9), 55))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)
        candidates = []
        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if height < min_line_px or width > 8 or height < width * 8 or area < height * 0.45:
                continue
            candidates.append({
                "x": float(x + width / 2),
                "y1": float(y),
                "y2": float(y + height),
                "height": float(height),
            })

        recovered = []
        for candidate in candidates:
            nearby_parallel = 0
            for other in candidates:
                if other is candidate:
                    continue
                if abs(other["x"] - candidate["x"]) > max(12.0, 0.60 * pixels_per_meter):
                    continue
                overlap = max(0.0, min(candidate["y2"], other["y2"]) - max(candidate["y1"], other["y1"]))
                shorter = max(1.0, min(candidate["height"], other["height"]))
                if overlap / shorter >= 0.55:
                    nearby_parallel += 1

            if nearby_parallel >= 3:
                continue

            recovered.append((candidate["x"], candidate["y1"], candidate["x"], candidate["y2"]))

        return recovered

    def _detect_filled_wall_centerlines(self, image_bgr: np.ndarray, pixels_per_meter: float,
                                        min_wall_length_m: float) -> List[Tuple]:
        """Recover centerlines from filled dark wall bands.

        Distance transform removes thin furniture outlines before directional
        morphology extracts the horizontal and vertical cores of thick walls.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        very_dark = (gray < 105).astype(np.uint8)
        distance = cv2.distanceTransform(very_dark, cv2.DIST_L2, 5)
        core = (distance >= 2.4).astype(np.uint8) * 255

        min_line_px = max(10, int(min_wall_length_m * pixels_per_meter * 0.55))
        kernel_len = max(9, min(45, min_line_px))
        masks = {
            "h": cv2.morphologyEx(
                core, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
            ),
            "v": cv2.morphologyEx(
                core, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
            ),
        }

        recovered = []
        for axis, axis_mask in masks.items():
            count, _, stats, _ = cv2.connectedComponentsWithStats(axis_mask, connectivity=8)
            for label_id in range(1, count):
                x = float(stats[label_id, cv2.CC_STAT_LEFT])
                y = float(stats[label_id, cv2.CC_STAT_TOP])
                width = float(stats[label_id, cv2.CC_STAT_WIDTH])
                height = float(stats[label_id, cv2.CC_STAT_HEIGHT])
                area = float(stats[label_id, cv2.CC_STAT_AREA])

                if axis == "h":
                    if width < min_line_px or width < height * 2.2 or area < width * 0.8:
                        continue
                    recovered.append((x, y + height / 2, x + width, y + height / 2))
                else:
                    if height < min_line_px or height < width * 2.2 or area < height * 0.8:
                        continue
                    recovered.append((x + width / 2, y, x + width / 2, y + height))

        return recovered

    def _snap_to_axis(self, segments: List[Tuple], angle_tolerance: float = 5.0) -> List[Tuple]:
        """Snap nearly-horizontal/vertical lines to exact H/V orientation.
        
        Most floor plan walls are perfectly horizontal or vertical.
        This dramatically reduces duplicate detection by normalizing angles.
        """
        snapped = []
        for (x1, y1, x2, y2) in segments:
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            
            if angle < angle_tolerance:
                # Nearly horizontal -> make perfectly horizontal
                avg_y = (y1 + y2) / 2
                snapped.append((min(x1, x2), avg_y, max(x1, x2), avg_y))
            elif angle > (90 - angle_tolerance):
                # Nearly vertical -> make perfectly vertical
                avg_x = (x1 + x2) / 2
                snapped.append((avg_x, min(y1, y2), avg_x, max(y1, y2)))
            else:
                # Diagonal - keep as is
                snapped.append((x1, y1, x2, y2))
        
        return snapped

    def _merge_collinear_segments(self, segments: List[Tuple], 
                                   perp_threshold: float = 8.0,
                                   gap_threshold: float = 15.0) -> List[Tuple]:
        """Aggressively merge segments that are collinear (same line, may overlap).
        
        This is the key function that eliminates overlapping/duplicate walls.
        Groups lines by orientation (H/V/diagonal) and proximity, then merges
        each group into a single segment spanning the full extent.
        """
        if len(segments) < 2:
            return segments
        
        # Separate into horizontal, vertical, diagonal
        h_lines = []  # nearly horizontal
        v_lines = []  # nearly vertical
        d_lines = []  # diagonal
        
        for seg in segments:
            x1, y1, x2, y2 = seg
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < 10:
                h_lines.append(seg)
            elif angle > 80:
                v_lines.append(seg)
            else:
                d_lines.append(seg)
        
        merged = []
        
        # Merge horizontal lines: group by Y coordinate
        merged += self._merge_parallel_lines(h_lines, axis='h', 
                                              perp_threshold=perp_threshold,
                                              gap_threshold=gap_threshold)
        
        # Merge vertical lines: group by X coordinate
        merged += self._merge_parallel_lines(v_lines, axis='v',
                                              perp_threshold=perp_threshold,
                                              gap_threshold=gap_threshold)
        
        # Merge diagonal lines  
        merged += self._merge_diagonal_lines(d_lines, perp_threshold, gap_threshold)
        
        return merged

    def _merge_parallel_lines(self, lines: List[Tuple], axis: str,
                               perp_threshold: float, gap_threshold: float) -> List[Tuple]:
        """Merge parallel lines along the same axis."""
        if not lines:
            return []
        
        # Sort by the perpendicular coordinate
        if axis == 'h':
            # For horizontal lines, group by Y value
            lines.sort(key=lambda l: (l[1] + l[3]) / 2)
        else:
            # For vertical lines, group by X value
            lines.sort(key=lambda l: (l[0] + l[2]) / 2)
        
        # Group nearby lines
        groups = []
        current_group = [lines[0]]
        
        for i in range(1, len(lines)):
            prev = current_group[-1]
            curr = lines[i]
            
            if axis == 'h':
                perp_dist = abs((curr[1] + curr[3]) / 2 - (prev[1] + prev[3]) / 2)
            else:
                perp_dist = abs((curr[0] + curr[2]) / 2 - (prev[0] + prev[2]) / 2)
            
            if perp_dist <= perp_threshold:
                current_group.append(curr)
            else:
                groups.append(current_group)
                current_group = [curr]
        
        groups.append(current_group)
        
        # For each group, merge into contiguous segments
        result = []
        for group in groups:
            result += self._merge_group_into_segments(group, axis, gap_threshold)
        
        return result

    def _merge_group_into_segments(self, group: List[Tuple], axis: str, 
                                    gap_threshold: float) -> List[Tuple]:
        """Merge a group of collinear lines into contiguous wall segments."""
        if not group:
            return []
        
        if axis == 'h':
            # Average Y coordinate
            avg_perp = np.mean([(l[1] + l[3]) / 2 for l in group])
            # Get all spans along X axis
            spans = [(min(l[0], l[2]), max(l[0], l[2])) for l in group]
            merged_spans = self._merge_1d_spans(spans, gap_threshold)
            return [(s[0], avg_perp, s[1], avg_perp) for s in merged_spans]
        else:
            # Average X coordinate
            avg_perp = np.mean([(l[0] + l[2]) / 2 for l in group])
            # Get all spans along Y axis
            spans = [(min(l[1], l[3]), max(l[1], l[3])) for l in group]
            merged_spans = self._merge_1d_spans(spans, gap_threshold)
            return [(avg_perp, s[0], avg_perp, s[1]) for s in merged_spans]

    def _merge_1d_spans(self, spans: List[Tuple], gap: float) -> List[Tuple]:
        """Merge overlapping or close 1D intervals."""
        if not spans:
            return []
        
        spans.sort(key=lambda s: s[0])
        merged = [spans[0]]
        
        for start, end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + gap:
                # Overlapping or close enough - extend
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        
        return merged

    def _merge_diagonal_lines(self, lines: List[Tuple], 
                               perp_threshold: float, gap_threshold: float) -> List[Tuple]:
        """Merge diagonal lines that are collinear."""
        if len(lines) < 2:
            return lines
        
        used = [False] * len(lines)
        result = []
        
        for i in range(len(lines)):
            if used[i]:
                continue
            
            group = [lines[i]]
            used[i] = True
            
            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue
                if (
                    self._are_segments_collinear(lines[i], lines[j], perp_threshold)
                    and self._segments_close_along_line(lines[i], lines[j], gap_threshold)
                ):
                    group.append(lines[j])
                    used[j] = True
            
            # Merge group: find the two most distant endpoints
            all_pts = []
            for seg in group:
                all_pts.append(np.array([seg[0], seg[1]]))
                all_pts.append(np.array([seg[2], seg[3]]))
            
            max_d = 0
            pa, pb = all_pts[0], all_pts[-1]
            for k in range(len(all_pts)):
                for l in range(k + 1, len(all_pts)):
                    d = np.linalg.norm(all_pts[k] - all_pts[l])
                    if d > max_d:
                        max_d = d
                        pa, pb = all_pts[k], all_pts[l]
            
            result.append((pa[0], pa[1], pb[0], pb[1]))
        
        return result

    def _are_segments_collinear(self, seg1: Tuple, seg2: Tuple, threshold: float) -> bool:
        """Check if two line segments are approximately collinear."""
        x1, y1, x2, y2 = seg1
        x3, y3, x4, y4 = seg2
        
        # Check angle similarity
        a1 = np.arctan2(y2 - y1, x2 - x1) % np.pi
        a2 = np.arctan2(y4 - y3, x4 - x3) % np.pi
        angle_diff = min(abs(a1 - a2), np.pi - abs(a1 - a2))
        
        if angle_diff > np.radians(10):
            return False
        
        # Check perpendicular distance of midpoint of seg2 to line of seg1
        mx, my = (x3 + x4) / 2, (y3 + y4) / 2
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-6:
            return np.sqrt((mx - x1)**2 + (my - y1)**2) < threshold
        
        dist = abs(dy * mx - dx * my + x2 * y1 - y2 * x1) / np.sqrt(length_sq)
        return dist < threshold

    def _segments_close_along_line(self, seg1: Tuple, seg2: Tuple, gap_threshold: float) -> bool:
        """Return True when collinear segments overlap or have only a small gap."""
        x1, y1, x2, y2 = seg1
        x3, y3, x4, y4 = seg2
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return False

        ux, uy = dx / length, dy / length
        span1 = sorted([0.0, length])
        p3 = (x3 - x1) * ux + (y3 - y1) * uy
        p4 = (x4 - x1) * ux + (y4 - y1) * uy
        span2 = sorted([p3, p4])
        return span2[0] <= span1[1] + gap_threshold and span1[0] <= span2[1] + gap_threshold

    def _sample_mask_support(self, mask: np.ndarray, seg: Tuple, sample_step: float = 8.0) -> float:
        """Measure how much of a segment is supported by a binary mask."""
        x1, y1, x2, y2 = seg
        h, w = mask.shape[:2]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        n_samples = max(5, int(length / sample_step))
        hits = 0

        for t in np.linspace(0, 1, n_samples):
            px = int(max(0, min(w - 1, round(x1 + t * (x2 - x1)))))
            py = int(max(0, min(h - 1, round(y1 + t * (y2 - y1)))))
            if mask[py, px] > 0:
                hits += 1

        return hits / n_samples

    def _segment_dark_support(self, gray: np.ndarray, seg: Tuple, sample_step: float = 8.0) -> float:
        """Measure whether a segment lies on dark/grey plan ink."""
        x1, y1, x2, y2 = seg
        h, w = gray.shape[:2]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        n_samples = max(5, int(length / sample_step))
        hits = 0

        for t in np.linspace(0, 1, n_samples):
            px = int(max(0, min(w - 1, round(x1 + t * (x2 - x1)))))
            py = int(max(0, min(h - 1, round(y1 + t * (y2 - y1)))))
            if gray[py, px] < 215:
                hits += 1

        return hits / n_samples

    def _filter_supported_segments(self, segments: List[Tuple], image_bgr: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> List[Tuple]:
        """Keep line candidates supported by either AI mask or visible plan ink."""
        if not segments:
            return []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        filtered = []

        for seg in segments:
            x1, y1, x2, y2 = seg
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 2:
                continue

            ai_support = self._sample_mask_support(mask, seg) if mask is not None else 0.0
            dark_support = self._segment_dark_support(gray, seg)

            if ai_support >= 0.18 or dark_support >= 0.32:
                filtered.append(seg)

        return filtered

    def _segment_thickness_px(self, gray: np.ndarray, seg: Tuple,
                              search_radius: int) -> float:
        """Estimate the median dark-band thickness perpendicular to a segment."""
        x1, y1, x2, y2 = seg
        h, w = gray.shape[:2]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if 10 <= angle <= 80:
            return 1.0

        thicknesses = []
        for t in np.linspace(0.15, 0.85, 9):
            px = int(round(x1 + t * (x2 - x1)))
            py = int(round(y1 + t * (y2 - y1)))
            if not (0 <= px < w and 0 <= py < h):
                continue

            if angle < 10:
                lo, hi = max(0, py - search_radius), min(h - 1, py + search_radius)
                values = gray[lo:hi + 1, px]
                anchor = py - lo
            else:
                lo, hi = max(0, px - search_radius), min(w - 1, px + search_radius)
                values = gray[py, lo:hi + 1]
                anchor = px - lo

            dark = values < 175
            runs = []
            start = None
            for idx, value in enumerate(dark):
                if value and start is None:
                    start = idx
                if start is not None and (not value or idx == len(dark) - 1):
                    end = idx if value and idx == len(dark) - 1 else idx - 1
                    runs.append((start, end))
                    start = None

            if not runs:
                continue
            nearest = min(runs, key=lambda run: 0 if run[0] <= anchor <= run[1]
                          else min(abs(anchor - run[0]), abs(anchor - run[1])))
            if (nearest[0] <= anchor <= nearest[1]
                    or min(abs(anchor - nearest[0]), abs(anchor - nearest[1]))
                    <= max(2, search_radius // 3)):
                thicknesses.append(nearest[1] - nearest[0] + 1)

        return float(np.median(thicknesses)) if thicknesses else 1.0

    def _sample_thick_ink_support(self, dark_distance: np.ndarray, seg: Tuple,
                                   search_radius: int) -> float:
        """Measure support from genuinely filled dark bands, not thin outlines."""
        x1, y1, x2, y2 = seg
        h, w = dark_distance.shape[:2]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        length = float(np.hypot(x2 - x1, y2 - y1))
        hits = 0
        samples = max(7, int(length / 10))

        for t in np.linspace(0.08, 0.92, samples):
            px = int(round(x1 + t * (x2 - x1)))
            py = int(round(y1 + t * (y2 - y1)))
            if angle < 10:
                values = dark_distance[max(0, py - search_radius):min(h, py + search_radius + 1),
                                       max(0, min(w - 1, px))]
            elif angle > 80:
                values = dark_distance[max(0, min(h - 1, py)),
                                       max(0, px - search_radius):min(w, px + search_radius + 1)]
            else:
                values = dark_distance[max(0, py - search_radius):min(h, py + search_radius + 1),
                                       max(0, px - search_radius):min(w, px + search_radius + 1)]
            if values.size and float(np.max(values)) >= 2.5:
                hits += 1

        return hits / samples

    @staticmethod
    def _point_segment_distance(point: Tuple[float, float], seg: Tuple) -> float:
        px, py = point
        x1, y1, x2, y2 = seg
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        if denom < 1e-9:
            return float(np.hypot(px - x1, py - y1))
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
        return float(np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))

    def _parallel_repeat_count(self, seg: Tuple, segments: List[Tuple],
                               pixels_per_meter: float) -> int:
        """Count axis-aligned segments stacked parallel to ``seg`` (hatch/stairs cue)."""
        x1, y1, x2, y2 = seg
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        seg_len = max(1.0, np.hypot(x2 - x1, y2 - y1))
        repeats = 0
        for other in segments:
            if other is seg:
                continue
            ox1, oy1, ox2, oy2 = other
            other_angle = np.degrees(np.arctan2(abs(oy2 - oy1), abs(ox2 - ox1)))
            if abs(angle - other_angle) > 6:
                continue
            if angle < 10:
                perpendicular_gap = abs((y1 + y2 - oy1 - oy2) / 2)
                overlap = max(0.0, min(max(x1, x2), max(ox1, ox2))
                              - max(min(x1, x2), min(ox1, ox2)))
            elif angle > 80:
                perpendicular_gap = abs((x1 + x2 - ox1 - ox2) / 2)
                overlap = max(0.0, min(max(y1, y2), max(oy1, oy2))
                              - max(min(y1, y2), min(oy1, oy2)))
            else:
                continue
            if perpendicular_gap <= 0.8 * pixels_per_meter and overlap / seg_len >= 0.55:
                repeats += 1
        return repeats

    def _filter_structural_network(self, segments: List[Tuple], image_bgr: np.ndarray,
                                   pixels_per_meter: float,
                                   mask: Optional[np.ndarray] = None) -> Tuple[List[Tuple], int]:
        """Reject isolated furniture strokes while preserving structural wall networks."""
        if not segments:
            return [], 0

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        search_radius = max(5, min(24, int(0.28 * pixels_per_meter)))
        join_distance = max(5.0, 0.24 * pixels_per_meter)
        lengths = [
            float(np.hypot(seg[2] - seg[0], seg[3] - seg[1])) / pixels_per_meter
            for seg in segments
        ]
        thicknesses = [
            self._segment_thickness_px(gray, seg, search_radius)
            for seg in segments
        ]
        ai_support = [
            self._sample_mask_support(mask, seg) if mask is not None else 0.0
            for seg in segments
        ]
        thick_threshold = max(4.0, 0.065 * pixels_per_meter)
        very_dark = (gray < 100).astype(np.uint8)
        dark_distance = cv2.distanceTransform(very_dark, cv2.DIST_L2, 5)
        thick_ink_support = [
            self._sample_thick_ink_support(
                dark_distance, seg, max(3, min(8, int(0.12 * pixels_per_meter)))
            )
            for seg in segments
        ]
        filled_wall_ink_ratio = float(np.mean(dark_distance >= 3.0))
        strong_wall_style = (
            sum(thickness >= thick_threshold for thickness in thicknesses)
            / max(1, len(thicknesses))
        ) >= 0.18 or filled_wall_ink_ratio >= 0.004

        adjacency = [set() for _ in segments]
        for i, seg in enumerate(segments):
            endpoints_i = ((seg[0], seg[1]), (seg[2], seg[3]))
            for j in range(i + 1, len(segments)):
                other = segments[j]
                endpoints_j = ((other[0], other[1]), (other[2], other[3]))
                connected = any(self._point_segment_distance(point, other) <= join_distance
                                for point in endpoints_i)
                connected = connected or any(self._point_segment_distance(point, seg) <= join_distance
                                              for point in endpoints_j)
                if connected:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        components = []
        unseen = set(range(len(segments)))
        while unseen:
            start = unseen.pop()
            component = {start}
            stack = [start]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(component)

        # A short partition stub that touches a long wall is part of the wall
        # network (e.g. a room divider joining a corridor), not stray furniture.
        long_wall_threshold_m = 2.4
        connected_to_long_wall = [False] * len(segments)
        for idx in range(len(segments)):
            for neighbor in adjacency[idx]:
                if lengths[neighbor] >= long_wall_threshold_m:
                    connected_to_long_wall[idx] = True
                    break

        keep_indexes = set()
        for component in components:
            xs = [coord for idx in component for coord in (segments[idx][0], segments[idx][2])]
            ys = [coord for idx in component for coord in (segments[idx][1], segments[idx][3])]
            span_m = max(max(xs) - min(xs), max(ys) - min(ys)) / pixels_per_meter
            total_m = sum(lengths[idx] for idx in component)
            max_length_m = max(lengths[idx] for idx in component)
            # A clean perpendicular corner (an H wall meeting a V wall) is a
            # strong architectural cue. Furniture rarely forms axis-aligned
            # L/T/+ junctions, so even a short such component is a real wall.
            junction_tol = max(6.0, 0.18 * pixels_per_meter)
            h_segs, v_segs = [], []
            for idx in component:
                seg = segments[idx]
                ang = np.degrees(np.arctan2(abs(seg[3] - seg[1]), abs(seg[2] - seg[0])))
                # Skip hatch/stair strokes so they cannot fake a wall junction.
                if self._parallel_repeat_count(seg, segments, pixels_per_meter) >= 3:
                    continue
                if ang < 10 and lengths[idx] >= 0.6:
                    h_segs.append(seg)
                elif ang > 80 and lengths[idx] >= 0.6:
                    v_segs.append(seg)
            # Require a true T/+ crossing where the meeting point lies in the
            # interior of at least one wall. End-to-end L-corners (as formed by a
            # small closed furniture rectangle) are deliberately excluded.
            interior_margin = max(8.0, 0.25 * pixels_per_meter)
            has_corner_junction = False
            for hs in h_segs:
                hy = (hs[1] + hs[3]) / 2
                hx_lo, hx_hi = sorted((hs[0], hs[2]))
                for vs in v_segs:
                    vx = (vs[0] + vs[2]) / 2
                    vy_lo, vy_hi = sorted((vs[1], vs[3]))
                    touches = (hx_lo - junction_tol <= vx <= hx_hi + junction_tol
                               and vy_lo - junction_tol <= hy <= vy_hi + junction_tol)
                    vx_interior_of_h = hx_lo + interior_margin <= vx <= hx_hi - interior_margin
                    hy_interior_of_v = vy_lo + interior_margin <= hy <= vy_hi - interior_margin
                    if touches and (vx_interior_of_h or hy_interior_of_v):
                        has_corner_junction = True
                        break
                if has_corner_junction:
                    break

            structural_component = (
                max_length_m >= 2.4
                or has_corner_junction
                or (span_m >= 2.8 and total_m >= 4.5)
                or (len(component) >= 3 and span_m >= 3.0 and total_m >= 5.0)
            )

            for idx in component:
                visibly_thick = (
                    thicknesses[idx] >= thick_threshold
                    and thick_ink_support[idx] >= 0.42
                )
                model_supported = ai_support[idx] >= 0.28
                seg = segments[idx]
                angle = np.degrees(np.arctan2(abs(seg[3] - seg[1]), abs(seg[2] - seg[0])))
                repeated_parallel = 0
                if lengths[idx] <= 2.5:
                    for other_idx, other in enumerate(segments):
                        if other_idx == idx:
                            continue
                        other_angle = np.degrees(np.arctan2(
                            abs(other[3] - other[1]), abs(other[2] - other[0])
                        ))
                        if abs(angle - other_angle) > 6:
                            continue
                        if angle < 10:
                            perpendicular_gap = abs((seg[1] + seg[3] - other[1] - other[3]) / 2)
                            overlap = max(0.0, min(max(seg[0], seg[2]), max(other[0], other[2]))
                                          - max(min(seg[0], seg[2]), min(other[0], other[2])))
                        elif angle > 80:
                            perpendicular_gap = abs((seg[0] + seg[2] - other[0] - other[2]) / 2)
                            overlap = max(0.0, min(max(seg[1], seg[3]), max(other[1], other[3]))
                                          - max(min(seg[1], seg[3]), min(other[1], other[3])))
                        else:
                            continue
                        if (perpendicular_gap <= 0.8 * pixels_per_meter
                                and overlap / max(1.0, lengths[idx] * pixels_per_meter) >= 0.55):
                            repeated_parallel += 1

                likely_hatch_or_stairs = repeated_parallel >= 3 and not visibly_thick
                component_support = structural_component
                # Short, thin partition stubs (room dividers) sit on dark ink and
                # are axis-aligned where they join a longer wall. Recover them even
                # on thick-wall plans, where they would otherwise be dropped.
                partition_stub = (
                    structural_component
                    and connected_to_long_wall[idx]
                    and (angle < 10 or angle > 80)
                    and self._segment_dark_support(gray, seg) >= 0.45
                    and lengths[idx] >= 0.6
                )
                # Axis-aligned members of a real perpendicular corner (e.g. the
                # short walls capping a corridor next to a stairwell) survive even
                # when the corner itself is too small to pass the size thresholds.
                junction_member = (
                    has_corner_junction
                    and (angle < 10 or angle > 80)
                    and self._segment_dark_support(gray, seg) >= 0.45
                    and lengths[idx] >= 0.6
                )
                if strong_wall_style:
                    # On plans drawn with filled/thick walls, connected furniture
                    # outlines are still not walls. Only unusually long thin runs
                    # get a structural fallback.
                    component_support = structural_component and lengths[idx] >= 3.5
                    if 10 <= angle <= 80:
                        component_support = False

                if not likely_hatch_or_stairs and (
                    visibly_thick or model_supported or component_support
                    or partition_stub or junction_member
                ):
                    keep_indexes.add(idx)

        kept = [seg for idx, seg in enumerate(segments) if idx in keep_indexes]
        return kept, len(segments) - len(kept)

    def _wall_metadata(self, image_bgr: np.ndarray, seg: Tuple,
                       pixels_per_meter: float) -> Dict:
        """Infer only material cues that are defensible from visual line thickness."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        thickness_px = self._segment_thickness_px(
            gray, seg, max(5, min(24, int(0.28 * pixels_per_meter)))
        )
        visual_thickness_mm = thickness_px / pixels_per_meter * 1000.0

        if visual_thickness_mm >= 180:
            wall_type = "thick_brick"
            material_confidence = 0.82
        elif visual_thickness_mm >= 75:
            wall_type = "common_brick"
            material_confidence = 0.68
        else:
            wall_type = None
            material_confidence = 0.0

        return {
            "wall_type": wall_type,
            "visual_thickness_px": float(round(thickness_px, 1)),
            "visual_thickness_mm": float(round(visual_thickness_mm, 1)),
            "material_confidence": material_confidence,
        }

    def _remove_duplicate_segments(self, segments: List[Tuple], perp_threshold: float,
                                   overlap_threshold: float = 0.72) -> List[Tuple]:
        """Remove near-identical wall segments after all merging/splitting."""
        if len(segments) < 2:
            return segments

        normalized = self._snap_to_axis(segments, angle_tolerance=8.0)
        buckets = {'h': [], 'v': [], 'd': []}

        for seg in normalized:
            x1, y1, x2, y2 = seg
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 1:
                continue

            if angle < 10:
                perp = (y1 + y2) / 2
                span = (min(x1, x2), max(x1, x2))
                buckets['h'].append((perp, span, seg, length))
            elif angle > 80:
                perp = (x1 + x2) / 2
                span = (min(y1, y2), max(y1, y2))
                buckets['v'].append((perp, span, seg, length))
            else:
                buckets['d'].append(seg)

        kept = []
        for axis in ('h', 'v'):
            lines = sorted(buckets[axis], key=lambda item: item[3], reverse=True)
            axis_kept = []
            for perp, span, seg, length in lines:
                duplicate = False
                for kept_perp, kept_span, _, kept_length in axis_kept:
                    overlap = max(0.0, min(span[1], kept_span[1]) - max(span[0], kept_span[0]))
                    shorter = max(1.0, min(length, kept_length))
                    if abs(perp - kept_perp) <= perp_threshold and overlap / shorter >= overlap_threshold:
                        duplicate = True
                        break

                if not duplicate:
                    axis_kept.append((perp, span, seg, length))

            kept.extend(item[2] for item in axis_kept)

        kept.extend(buckets['d'])
        return kept

    def _make_wall_ink_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Build a mask of likely wall ink from the original plan image."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        dark = cv2.inRange(gray, 0, 190)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        ink = cv2.bitwise_or(dark, adaptive)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel_small, iterations=1)
        return ink

    def _best_supported_axis_coordinate(self, ink: np.ndarray, axis: str, fixed: float,
                                        start: float, end: float, search_radius: int) -> Optional[float]:
        """Snap a horizontal/vertical candidate to the nearest row/column with wall ink."""
        h, w = ink.shape[:2]
        fixed_i = int(round(fixed))
        start_i = int(max(0, round(min(start, end))))
        end_i = int(round(max(start, end)))

        if axis == 'h':
            start_i = min(start_i, w - 1)
            end_i = min(max(start_i + 1, end_i), w - 1)
            lo = max(0, fixed_i - search_radius)
            hi = min(h - 1, fixed_i + search_radius)
        else:
            start_i = min(start_i, h - 1)
            end_i = min(max(start_i + 1, end_i), h - 1)
            lo = max(0, fixed_i - search_radius)
            hi = min(w - 1, fixed_i + search_radius)

        best_coord = None
        best_score = 0.0
        for coord in range(lo, hi + 1):
            if axis == 'h':
                score = float(np.count_nonzero(ink[coord, start_i:end_i + 1]))
            else:
                score = float(np.count_nonzero(ink[start_i:end_i + 1, coord]))

            # Prefer the closest row/column when support is similar.
            score -= abs(coord - fixed_i) * 0.12
            if score > best_score:
                best_score = score
                best_coord = coord

        min_expected_support = max(4, (end_i - start_i) * 0.04)
        if best_coord is None or best_score < min_expected_support:
            return None

        return float(best_coord)

    def _supported_runs_for_axis_segment(self, ink: np.ndarray, axis: str, fixed: float,
                                         start: float, end: float, band_radius: int,
                                         max_gap_px: int, min_run_px: float) -> List[Tuple[float, float]]:
        """Split an axis-aligned segment into runs that are actually backed by ink."""
        h, w = ink.shape[:2]
        fixed_i = int(round(fixed))
        start_i = int(max(0, round(min(start, end))))
        end_i = int(round(max(start, end)))

        if axis == 'h':
            start_i = min(start_i, w - 1)
            end_i = min(max(start_i + 1, end_i), w - 1)
            lo = max(0, fixed_i - band_radius)
            hi = min(h - 1, fixed_i + band_radius)
            support = np.any(ink[lo:hi + 1, start_i:end_i + 1] > 0, axis=0)
        else:
            start_i = min(start_i, h - 1)
            end_i = min(max(start_i + 1, end_i), h - 1)
            lo = max(0, fixed_i - band_radius)
            hi = min(w - 1, fixed_i + band_radius)
            support = np.any(ink[start_i:end_i + 1, lo:hi + 1] > 0, axis=1)

        runs = []
        run_start = None
        last_support = None

        for idx, has_support in enumerate(support):
            pos = start_i + idx
            if has_support:
                if run_start is None:
                    run_start = pos
                last_support = pos
            elif run_start is not None and last_support is not None and pos - last_support > max_gap_px:
                if last_support - run_start >= min_run_px:
                    runs.append((float(run_start), float(last_support)))
                run_start = None
                last_support = None

        if run_start is not None and last_support is not None and last_support - run_start >= min_run_px:
            runs.append((float(run_start), float(last_support)))

        return runs

    def _refine_segments_to_ink(self, segments: List[Tuple], image_bgr: np.ndarray,
                                pixels_per_meter: float, min_wall_length_m: float) -> List[Tuple]:
        """Snap merged segments back to floor-plan ink and cut unsupported spans."""
        if not segments:
            return []

        ink = self._make_wall_ink_mask(image_bgr)
        min_run_px = max(8.0, min_wall_length_m * pixels_per_meter)
        search_radius = max(4, min(14, int(0.18 * pixels_per_meter)))
        band_radius = max(2, min(7, int(0.08 * pixels_per_meter)))
        max_gap_px = max(4, min(12, int(0.18 * pixels_per_meter)))
        refined = []

        for seg in segments:
            x1, y1, x2, y2 = seg
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

            if angle < 10:
                start, end = sorted((x1, x2))
                fixed = self._best_supported_axis_coordinate(
                    ink, axis='h', fixed=(y1 + y2) / 2,
                    start=start, end=end, search_radius=search_radius
                )
                if fixed is None:
                    continue

                runs = self._supported_runs_for_axis_segment(
                    ink, axis='h', fixed=fixed, start=start, end=end,
                    band_radius=band_radius, max_gap_px=max_gap_px, min_run_px=min_run_px
                )
                refined.extend((run_start, fixed, run_end, fixed) for run_start, run_end in runs)

            elif angle > 80:
                start, end = sorted((y1, y2))
                fixed = self._best_supported_axis_coordinate(
                    ink, axis='v', fixed=(x1 + x2) / 2,
                    start=start, end=end, search_radius=search_radius
                )
                if fixed is None:
                    continue

                runs = self._supported_runs_for_axis_segment(
                    ink, axis='v', fixed=fixed, start=start, end=end,
                    band_radius=band_radius, max_gap_px=max_gap_px, min_run_px=min_run_px
                )
                refined.extend((fixed, run_start, fixed, run_end) for run_start, run_end in runs)

            else:
                if self._segment_dark_support(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), seg) >= 0.55:
                    refined.append(seg)

        return refined

    def _extract_walls_from_mask(self, mask: np.ndarray, pixels_per_meter: float,
                                   min_wall_length_m: float = 0.5) -> List[Tuple]:
        """Extract wall line segments from a binary segmentation mask.
        
        Uses skeletonization to find the center lines of walls, then
        extracts line segments using connected component analysis + PCA.
        """
        h, w = mask.shape[:2]
        min_wall_px = min_wall_length_m * pixels_per_meter
        
        # Skeletonize the mask to get 1px center lines
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask > 0).astype(np.uint8) * 255
        
        # Find connected components in skeleton
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        
        segments = []
        for label_id in range(1, num_labels):  # Skip background (0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_wall_px * 0.3:  # Too small to be a wall
                continue
            
            # Get points belonging to this component
            pts = np.column_stack(np.where(labels == label_id))  # (row, col) format
            if len(pts) < 2:
                continue
            
            # Use PCA to find the principal axis
            mean = pts.mean(axis=0)
            centered = pts - mean
            cov = np.cov(centered.T)
            
            if cov.shape != (2, 2):
                continue
                
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # The principal direction is the eigenvector with largest eigenvalue
            principal = eigenvectors[:, 1]  # (row_dir, col_dir)
            
            # Project all points onto the principal axis
            projections = centered @ principal
            
            # Check linearity: ratio of eigenvalues
            # High ratio = very linear (wall-like), low ratio = blobby
            linearity = eigenvalues[1] / (eigenvalues[0] + 1e-6)
            if linearity < 3.0:  # Not linear enough
                continue
            
            # Get endpoints by finding min/max projections
            min_proj_idx = np.argmin(projections)
            max_proj_idx = np.argmax(projections)
            
            p1 = pts[min_proj_idx]  # (row, col)
            p2 = pts[max_proj_idx]  # (row, col)
            
            # Convert to (x, y) = (col, row) format
            x1, y1 = float(p1[1]), float(p1[0])
            x2, y2 = float(p2[1]), float(p2[0])
            
            # Check length
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length < min_wall_px:
                continue
            
            segments.append((x1, y1, x2, y2))
        
        return segments

    def _split_at_intersections(self, segments: List[Tuple], 
                                  tolerance: float = 10.0) -> List[Tuple]:
        """Split long segments at T-junction / cross-intersection points.
        
        This prevents a single wall from spanning across room boundaries.
        """
        if len(segments) < 2:
            return segments
        
        result = []
        for i, seg in enumerate(segments):
            x1, y1, x2, y2 = seg
            split_points = []
            
            for j, other in enumerate(segments):
                if i == j:
                    continue
                ox1, oy1, ox2, oy2 = other
                
                # Check if any endpoint of 'other' is close to the middle of 'seg'
                for px, py in [(ox1, oy1), (ox2, oy2)]:
                    # Project point onto segment
                    dx, dy = x2 - x1, y2 - y1
                    seg_len = np.sqrt(dx*dx + dy*dy)
                    if seg_len < 1:
                        continue
                    
                    t = ((px - x1) * dx + (py - y1) * dy) / (seg_len * seg_len)
                    
                    # Only split if point is in the interior (not near endpoints)
                    if 0.1 < t < 0.9:
                        # Distance from point to line
                        proj_x = x1 + t * dx
                        proj_y = y1 + t * dy
                        dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
                        
                        if dist < tolerance:
                            split_points.append(t)
            
            if split_points:
                # Sort split points and create sub-segments
                split_points = sorted(set([0.0] + split_points + [1.0]))
                for k in range(len(split_points) - 1):
                    t0, t1 = split_points[k], split_points[k+1]
                    sx1 = x1 + t0 * (x2 - x1)
                    sy1 = y1 + t0 * (y2 - y1)
                    sx2 = x1 + t1 * (x2 - x1)
                    sy2 = y1 + t1 * (y2 - y1)
                    result.append((sx1, sy1, sx2, sy2))
            else:
                result.append(seg)
        
        return result

    def detect_from_base64(
        self, 
        image_b64: str, 
        pixels_per_meter: float,
        threshold: float = 0.5,
        min_wall_length_m: float = 0.5
    ) -> Dict:
        """Full pipeline: base64 image → detected walls.
        
        Strategy (hybrid):
        1. If AI model available: get segmentation mask → skeleton → line segments
        2. Always run classical CV/morphology to recover thin missed walls
        3. Snap to axes, merge collinear, split at intersections, suppress duplicates
        """
        try:
            # Decode base64
            if ',' in image_b64:
                _, encoded = image_b64.split(',', 1)
            else:
                encoded = image_b64
            
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"success": False, "walls": [], "message": "Failed to decode image"}
            
            h, w = image.shape[:2]
            min_wall_px = min_wall_length_m * pixels_per_meter
            
            raw_segments = []
            mask = None
            
            # === AI segmentation ===
            if self.has_model:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = self.predict_mask(image_rgb, threshold=threshold)
                
                if mask.sum() > 100:
                    # Extract walls from AI mask using skeleton
                    raw_segments = self._extract_walls_from_mask(mask, pixels_per_meter, min_wall_length_m)
                    print(f"[AutoTrace] AI skeleton segments: {len(raw_segments)}")
            
            # === Classical CV companion pass ===
            # Always run this pass. AI can find enough walls overall while still
            # missing long thin walls, so fallback-only behavior leaves gaps.
            classical = self.detect_lines_classical(image, pixels_per_meter, min_wall_length_m)
            classical = self._filter_supported_segments(classical, image, mask)
            print(f"[AutoTrace] Classical CV supported segments: {len(classical)}")
            raw_segments.extend(classical)
            print(f"[AutoTrace] Combined raw segments: {len(raw_segments)}")

            if not raw_segments:
                return {
                    "success": True,
                    "walls": [],
                    "count": 0,
                    "image_size": {"width": w, "height": h},
                    "message": "No wall-like lines were detected"
                }
            
            # Step 1: Snap to axes
            snapped = self._snap_to_axis(raw_segments, angle_tolerance=7.0)
            print(f"[AutoTrace] After axis snapping: {len(snapped)}")
            
            # Step 2: Aggressive merge of collinear/parallel segments
            # Walls are typically 15-30cm thick, so lines within 30cm are the same wall
            perp_thresh = max(10.0, 0.30 * pixels_per_meter)  # ~30cm
            gap_thresh = max(15.0, 0.50 * pixels_per_meter)   # ~50cm gap
            
            merged = self._merge_collinear_segments(snapped, perp_thresh, gap_thresh)
            print(f"[AutoTrace] After first merge: {len(merged)}")
            
            # Second pass merge to catch any remaining duplicates
            merged = self._merge_collinear_segments(merged, perp_thresh, gap_thresh)
            print(f"[AutoTrace] After second merge: {len(merged)}")
            
            # Step 3: Split at intersections
            split = self._split_at_intersections(merged, tolerance=perp_thresh)
            print(f"[AutoTrace] After splitting at intersections: {len(split)}")
            
            # Step 4: Remove near-identical overlaps left by AI/CV agreement
            deduped = self._remove_duplicate_segments(split, perp_threshold=perp_thresh)
            print(f"[AutoTrace] After duplicate suppression: {len(deduped)}")

            # Step 5: Snap back to visible wall ink and cut lines through empty gaps
            refined = self._refine_segments_to_ink(
                deduped,
                image_bgr=image,
                pixels_per_meter=pixels_per_meter,
                min_wall_length_m=min_wall_length_m
            )
            print(f"[AutoTrace] After ink refinement: {len(refined)}")

            horizontal_recovered = self._detect_long_horizontal_wall_runs(
                image,
                pixels_per_meter=pixels_per_meter,
                min_wall_length_m=min_wall_length_m,
            )
            refined.extend(horizontal_recovered)
            print(f"[AutoTrace] Horizontal wall recovery added: {len(horizontal_recovered)}")

            vertical_recovered = self._detect_long_vertical_wall_runs(
                image,
                pixels_per_meter=pixels_per_meter,
                min_wall_length_m=min_wall_length_m,
            )
            refined.extend(vertical_recovered)
            print(f"[AutoTrace] Vertical wall recovery added: {len(vertical_recovered)}")

            refined = self._remove_duplicate_segments(refined, perp_threshold=max(5.0, 0.18 * pixels_per_meter))
            print(f"[AutoTrace] After refined duplicate suppression: {len(refined)}")

            refined, filtered_object_count = self._filter_structural_network(
                refined,
                image_bgr=image,
                pixels_per_meter=pixels_per_meter,
                mask=mask,
            )
            print(f"[AutoTrace] After furniture/object filtering: {len(refined)} "
                  f"(removed {filtered_object_count})")

            filled_wall_recovered = self._detect_filled_wall_centerlines(
                image,
                pixels_per_meter=pixels_per_meter,
                min_wall_length_m=min_wall_length_m,
            )
            refined.extend(filled_wall_recovered)
            refined = self._remove_duplicate_segments(
                refined, perp_threshold=max(5.0, 0.18 * pixels_per_meter)
            )
            print(f"[AutoTrace] Filled thick-wall recovery added: {len(filled_wall_recovered)}; "
                  f"combined count: {len(refined)}")
            
            # Step 6: Filter by minimum length
            walls = []
            for (x1, y1, x2, y2) in refined:
                length_px = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                length_m = length_px / pixels_per_meter
                
                if length_m < min_wall_length_m:
                    continue

                segment = (x1, y1, x2, y2)
                metadata = self._wall_metadata(image, segment, pixels_per_meter)
                walls.append({
                    "points": [
                        [float(x1 / pixels_per_meter), float(y1 / pixels_per_meter)],
                        [float(x2 / pixels_per_meter), float(y2 / pixels_per_meter)]
                    ],
                    "length_m": float(round(length_m, 2)),
                    "wall_type": metadata["wall_type"],
                    "type": "wall",
                    "confidence": 0.9,
                    **metadata,
                })
            
            walls.sort(key=lambda w: w["length_m"], reverse=True)
            print(f"[AutoTrace] Final wall count: {len(walls)}")
            
            return {
                "success": True,
                "walls": walls,
                "count": len(walls),
                "filtered_object_count": filtered_object_count,
                "image_size": {"width": w, "height": h},
                "message": f"Detected {len(walls)} wall segments"
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "walls": [], "message": f"Detection error: {str(e)}"}


# Singleton instance
_detector_instance = None

def get_detector(weights_path=None):
    """Get or create singleton WallDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = WallDetector(weights_path=weights_path)
    return _detector_instance
