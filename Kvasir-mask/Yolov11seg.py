import os
import sys
import json
import glob
import random
import shutil
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
print("Current working directory:", os.getcwd())
os.chdir("/home/luca/Desktop/Luca/File-di-kvasir") #metti come directory il path del progetto, all'interno del quale si trova la cartella kvasir-mask

# ========== CONFIGURATION ==========
IMAGE_DIR = "Kvasir-mask/images"          
MASK_DIR = "Kvasir-mask/masks"            
JSON_PATH = "Kvasir-mask/bounding-boxes.json" 
OUTPUT_DIR = "Kvasir-mask/kvasir_yolo_seg_dataset"
MODEL_SIZE = 'm'  
BATCH_SIZE = 16   
EPOCHS = 100
IMG_SIZE = 640
DATA_YAML = f"{OUTPUT_DIR}/data.yaml"

class KvasirToYOLOSeg:
    """Convert Kvasir masks + JSON to YOLO segmentation format with healthy images."""
    MIN_AREA = 200  # Minimum area to consider a contour a valid polyp
    MAX_ASPECT_RATIO = 8.0

    def __init__(self, image_dir, mask_dir, json_path, output_dir, seed=42):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.seed = seed

        # Load JSON annotations (for polyp images only)
        with open(self.json_path, 'r') as f:
            self.annotations = json.load(f)

        if not isinstance(self.annotations, dict):
            raise ValueError("Annotations JSON must be a dict keyed by image ID.")

    @staticmethod
    def mask_to_polygon(mask):
        

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        valid_polygons = []
        
        # Calculate the approximation tolerance (epsilon) based on the perimeter
        epsilon_multiplier = 0.001

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            epsilon = epsilon_multiplier * perimeter
            
            # Approximate the contour to simplify the polygon
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calculate metrics for noise filtering
            area = cv2.contourArea(approx)
            x, y, approx_w, approx_h = cv2.boundingRect(approx)
            
            # Avoid division by zero
            aspect_ratio = approx_w / approx_h if approx_h != 0 else KvasirToYOLOSeg.MAX_ASPECT_RATIO + 1
            
            # 1. Area Check: Filters out contours that are too small (noise)
            if area < KvasirToYOLOSeg.MIN_AREA:
                continue
            
            # 2. Filter out contours that are too thin/elongated (e.g., line artifacts)
            if aspect_ratio > KvasirToYOLOSeg.MAX_ASPECT_RATIO or 1/aspect_ratio > KvasirToYOLOSeg.MAX_ASPECT_RATIO:
                continue
            
            # Append the raw NumPy array coordinates
            valid_polygons.append(approx.reshape(-1, 2))
            
        return valid_polygons if valid_polygons else None


    @staticmethod
    def normalize_polygon(polygon, img_width, img_height):
        """Normalize polygon coordinates to [0, 1]."""
        polygon = polygon.astype(float)
        polygon[:, 0] /= img_width
        polygon[:, 1] /= img_height
        
        # Clamp to valid range
        polygon = np.clip(polygon, 0.0, 1.0)
        return polygon

    def prepare_dataset(self, train_split=0.7, val_split=0.2, test_split=0.1):
        """Prepare segmentation dataset with polyp + healthy images."""
        
        if abs(train_split + val_split + test_split - 1.0) > 1e-8:
            raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")

        # Create directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

        # Get all images
        all_images = list(self.image_dir.glob('*.jpg')) + list(self.image_dir.glob('*.png'))
        
        # Separate polyp vs healthy images
        polyp_images = [img for img in all_images if img.stem in self.annotations]
        healthy_images = [img for img in all_images if img.stem not in self.annotations]
        
        # **FIX: Shuffle BEFORE splitting to prevent data leakage**
        rnd = random.Random(self.seed)
        rnd.shuffle(polyp_images)
        rnd.shuffle(healthy_images)
        
        def split_list(lst, train_r, val_r):
            n = len(lst)
            n_train = int(n * train_r)
            n_val = int(n * val_r)
            return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

        polyp_train, polyp_val, polyp_test = split_list(polyp_images, train_split, val_split)
        healthy_train, healthy_val, healthy_test = split_list(healthy_images, train_split, val_split)

        splits = {
            'train': polyp_train + healthy_train,
            'val': polyp_val + healthy_val,
            'test': polyp_test + healthy_test
        }

        # Shuffle combined splits to mix polyp and healthy images
        for split_imgs in splits.values():
            rnd.shuffle(split_imgs)

        print(f"\n{'='*70}")
        print("DATASET SPLIT")
        print(f"Train: {len(splits['train'])} ({len(polyp_train)} polyps + {len(healthy_train)} healthy)")
        print(f"Val:   {len(splits['val'])} ({len(polyp_val)} polyps + {len(healthy_val)} healthy)")
        print(f"Test:  {len(splits['test'])} ({len(polyp_test)} polyps + {len(healthy_test)} healthy)")
        print(f"{'='*70}\n")

        # Process each split
        for split_name, images in splits.items():
            print(f"Processing {split_name} split ({len(images)} images)...")
            self._process_split(images, split_name)

        # Create YAML
        self._create_yaml()

        print(f"Segmentation dataset created!")
        print(f"  Output: {self.output_dir.resolve()}")


    def _process_split(self, image_files, split_name):
        img_dir = self.output_dir / 'images' / split_name
        label_dir = self.output_dir / 'labels' / split_name

        for img_path in image_files:
            img_id = img_path.stem

            # Load image and copy to output directory
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            h, w = img.shape[:2]
            shutil.copy(img_path, img_dir / img_path.name)

            label_path = label_dir / f"{img_id}.txt"

            if img_id in self.annotations:
                # Polyp image processing
                
                # Load and prepare mask
                mask_path = self.mask_dir / f"{img_id}.jpg"
                if not mask_path.exists():
                    mask_path = self.mask_dir / f"{img_id}.png"

                if not mask_path.exists():
                    print(f"Warning: Mask not found for {img_id}")
                    continue

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Could not read mask for {img_id}")
                    continue

                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                # Get the SINGLE clean polygon (from the mask_to_polygon logic)
                polygons = self.mask_to_polygon(mask)
                if not polygons:
                    print(f"Warning: No valid polygons for {img_id}. SKIPPING image.")
                    continue

                with open(label_path, 'w') as f:
                        for polygon in polygons:
                            norm_poly = self.normalize_polygon(polygon, w, h)
                            # Format for YOLO: class_id x1 y1 x2 y2 ...
                            coords = ' '.join(f"{x:.6f} {y:.6f}" for x, y in norm_poly)
                            f.write(f"0 {coords}\n")
            else:
                # Healthy image — create empty label file
                label_path.touch()
                
    def _create_yaml(self):
        """Create data.yaml for segmentation."""
        data = {
            'path': str(self.output_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['polyp']
        }

        yaml_path = self.output_dir / 'data.yaml'
        # NOTE: Assumes 'yaml' library is available
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"\n  Created: {yaml_path}")

def train_yolo_seg(data_yaml_path, model_size=MODEL_SIZE, epochs=100, img_size=640, 
                   batch_size=BATCH_SIZE, workers=4, lr0=1e-4):
    """Train YOLOv11 Segmentation model."""
    
    # Device detection
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 0
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = 'cpu'
        print("Using CPU")

    # Load YOLOv11-seg model
    model = YOLO(f'yolo11{model_size}-seg.pt')  # YOLOv11 segmentation

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='polyp_segmentation_v11',
        patience=20,
        save=True,
        device=device,
        workers=workers,
        optimizer='AdamW',
        project='Kvasir-mask',
        
        # Learning rate settings
        lr0=lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5,
        warmup_momentum=0.8,
        momentum=0.937,
        weight_decay=0.001,
        dropout=0.1,
        
        # Multi-scale training
        multi_scale=True,
        
        # Medical imaging augmentations
        mosaic=0.0,          # Disabled for medical
        mixup=0.0,           # Light mixup
        copy_paste=0.0,      # Copy-paste augmentation
        erasing=0.1,         # Random erasing
        hsv_h=0.01,          # Minimal hue (preserve color)
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.1,
        flipud=0.5,
        fliplr=0.5,
        shear=1.0,
        perspective=0.0001,
        
        # Advanced augmentations
        augment=True,
        auto_augment='randaugment',
        
        # Segmentation specific
        mask_ratio=4,
        overlap_mask=True
    )
    
    return model

def add_gt_overlay(img, label_path):
    """Add ground truth overlay to image (green boxes/masks)."""
    if not label_path.exists():
        return img
    
    h, w = img.shape[:2]
    overlay = img.copy()
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:  # Need at least class + 2 points
                continue
            
            # Parse polygon points
            coords = list(map(float, parts[1:]))
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                points.append([x, y])
            
            if len(points) >= 3:
                # Draw filled polygon (semi-transparent green)
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                # Draw polygon outline
                cv2.polylines(overlay, [pts], True, (0, 200, 0), 2)
    
    # Blend with original
    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    return img

def evaluate_model_seg(model_path, data_yaml_path, split_name, conf=0.001, iou=0.5):
  
    model = YOLO(model_path)
    
    print(f"\nRunning validation with Conf={conf}, IoU={iou} on split: {split_name}")
    metrics = model.val(
        data=data_yaml_path, 
        split=split_name,
        conf=conf,
        iou=iou
    )

    # --- CHECK IF SEGMENTATION METRICS EXIST ---
    if not hasattr(metrics, 'seg') or metrics.seg is None:
        print("\nERROR: This is not a segmentation model! metrics.seg is not available.")
        print("Make sure you're using a YOLO-seg model (e.g., yolo-seg.pt)")
        return metrics

    # --- GET YOLO'S REPORTED METRICS ---
    # These metrics are at the confidence threshold that gives the *optimal F1-score*
    # Box metrics
    box_p_yolo = metrics.box.p[0] if hasattr(metrics.box, 'p') and len(metrics.box.p) > 0 else 0.0
    box_r_yolo = metrics.box.r[0] if hasattr(metrics.box, 'r') and len(metrics.box.r) > 0 else 0.0
    box_map50 = metrics.box.map50 if hasattr(metrics.box, 'map50') else 0.0
    box_map = metrics.box.map if hasattr(metrics.box, 'map') else 0.0
    
    # Mask metrics
    mask_p_yolo = metrics.seg.p[0] if hasattr(metrics.seg, 'p') and len(metrics.seg.p) > 0 else 0.0
    mask_r_yolo = metrics.seg.r[0] if hasattr(metrics.seg, 'r') and len(metrics.seg.r) > 0 else 0.0
    mask_map50 = metrics.seg.map50 if hasattr(metrics.seg, 'map50') else 0.0
    mask_map = metrics.seg.map if hasattr(metrics.seg, 'map') else 0.0
    
    # --- Initialize variables for summary ---
    total_gt = 0
    total_pred = 0
    TP = 0
    FP = 0
    FN = 0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    # --- EXTRACT CONFUSION MATRIX ---
    # Note: The confusion matrix is calculated at the *specific conf* passed to model.val()
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        cm = metrics.confusion_matrix.matrix
        
        print(f"\nDEBUG - Confusion Matrix:")
        print(f"{cm}")
        print(f"\nStructure (for single-class detection):")
        print(f"  Rows = Predicted, Cols = Actual")
        print(f"  [[TP, FP],   ← Row 0: Predicted polyps")
        print(f"   [FN, TN]]   ← Row 1: Predicted background")
        print(f"\nInterpretation:")
        print(f"  cm[0,0] = {int(cm[0,0])}: True polyps correctly detected (TP)")
        print(f"  cm[0,1] = {int(cm[0,1])}: Background predicted as polyp (FP) <--- FALSE POSITIVE")
        print(f"  cm[1,0] = {int(cm[1,0])}: True polyps missed (FN)            <--- FALSE NEGATIVE")
        print(f"  cm[1,1] = {int(cm[1,1])}: Background correctly identified (TN)")
        
        # Extract TP/FP/FN
        TP = int(cm[0, 0])  # Predicted Polyp, Actual Polyp
        FP = int(cm[0, 1])  # Predicted Polyp, Actual Background
        FN = int(cm[1, 0])  # Predicted Background, Actual Polyp
        
        total_pred = TP + FP
        total_gt = TP + FN
        
        # Calculate metrics from confusion matrix
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\nCalculated from Confusion Matrix (conf={conf}):")
        print(f"  Total Ground Truth: {total_gt}")
        print(f"  Total Predictions:  {total_pred}")
        print(f"  TP={TP}, FP={FP}, FN={FN}")
        print(f"  Precision: {precision:.4f} = {TP}/({TP}+{FP})")
        print(f"  Recall:    {recall:.4f} = {TP}/({TP}+{FN})")
        print(f"  F1-score:  {f1:.4f}")
        
        # Sanity check against metrics.confusion_matrix.nt
        # nt[0] = number of targets for class 0 (polyps)
        if hasattr(metrics.confusion_matrix, 'nt') and len(metrics.confusion_matrix.nt) > 0:
             total_gt_yolo = int(metrics.confusion_matrix.nt[0])
             if total_gt_yolo != total_gt:
                 print(f"  WARNING: CM Total GT ({total_gt}) != metrics.nt[0] ({total_gt_yolo})")
             else:
                 print(f"  (Total Ground Truth {total_gt} matches YOLO's instance count)")

    else:
        print("\nWARNING: Confusion matrix not available!")
        # Fallback using YOLO's reported metrics (less precise)
        precision = box_p_yolo
        recall = box_r_yolo
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # We can't know TP/FP/FN for sure without the CM at this conf
        total_gt = 0 # Unknown
        

    # --- PRINT SUMMARY ---
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS ON {split_name.upper()} SET")
    print(f"Confidence: {conf} | IoU Threshold: {iou}")
    if total_gt > 0:
        print(f"Total Ground Truth Polyps: {total_gt}")
    print(f"{'='*70}")
    
    print("\n### Box (Detection) Metrics")
    print(f"YOLO Reported (at optimal F1 conf):")
    print(f"  Precision: {box_p_yolo:.4f}")
    print(f"  Recall:    {box_r_yolo:.4f}")
    print(f"  mAP@0.50:  {box_map50:.4f}")
    print(f"  mAP@0.50:0.95: {box_map:.4f}")
    
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        print(f"\nFrom Confusion Matrix (at conf={conf}):")
        print(f"  Total Predicted: {total_pred}")
        print(f"  TP/FP/FN: {TP}/{FP}/{FN}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
    
    print("\n### Mask (Segmentation) Metrics")
    print(f"YOLO Reported (at optimal F1 conf):")
    print(f"  Precision: {mask_p_yolo:.4f}")
    print(f"  Recall:    {mask_r_yolo:.4f}")
    print(f"  mAP@0.50:  {mask_map50:.4f}")
    print(f"  mAP@0.50:0.95: {mask_map:.4f}")
    
    if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
        print(f"\nFrom Confusion Matrix (at conf={conf}):")
        print(f"  Note: Same TP/FP/FN as box (YOLO uses single confusion matrix)")
        print(f"  Total Predicted: {total_pred}")
        print(f"  TP/FP/FN: {TP}/{FP}/{FN}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
    
    print(f"{'='*70}\n")

    return metrics

def predict_on_all_images_seg(model_path, image_dir, data_yaml_path=None, 
                               conf_threshold=0.25, save_dir='predictions_seg', iou=0.5):
    """
    Run inference, save predicted images with GT overlay, and compute statistics.
    """
    model = YOLO(model_path)
    image_dir = Path(image_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- SETUP PATHS ---
    # Assumes standard YOLO dataset structure: .../dataset/images/test, .../dataset/labels/test
    label_dir = image_dir.parent.parent / 'labels' / image_dir.name
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # --- INITIALIZE COUNTERS ---
    total_gt_polyps = 0
    total_pred_polyps = 0
    TP_img = FP_img = FN_img = TN_img = 0

    print(f"\n{'='*70}")
    print(f"1. RUNNING INFERENCE AND VISUALIZATION ON {len(image_files)} IMAGES (Saving GT Overlay)")
    print(f"   Conf={conf_threshold}, IoU={iou}")
    print(f"{'='*70}\n")
    
    # --- 1. INFERENCE, VISUALIZATION, AND IMAGE-LEVEL COUNTING LOOP ---
    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"
        
        # Check ground truth
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                gt_boxes = [line for line in f if line.strip()]
        num_gt = len(gt_boxes)
        total_gt_polyps += num_gt

        # Run inference (single image)
        results = model(str(img_path), conf=conf_threshold, iou=iou, verbose=False)
        result = results[0] 

        img_with_results = result.plot()

        img_with_gt_overlay = add_gt_overlay(img_with_results, label_path)
        
        # 3. Save the final image
        output_path = save_dir / f"pred_gt_{img_path.name}"

        # img_with_gt_overlay is ALREADY in BGR format, which cv2.imwrite needs.
        img_to_save = np.ascontiguousarray(img_with_gt_overlay, dtype=np.uint8)
        cv2.imwrite(str(output_path), img_to_save)
        # Count predictions
        num_pred = 0
        if result.masks is not None:
            num_pred = len(result.masks)
        
        total_pred_polyps += num_pred
        
        # Contingency matrix (image-level)
        if num_gt > 0 and num_pred > 0:
            TP_img += 1
        elif num_gt == 0 and num_pred > 0:
            FP_img += 1
        elif num_gt > 0 and num_pred == 0:
            FN_img += 1
        elif num_gt == 0 and num_pred == 0:
            TN_img += 1
            
        print(f"  Processed {img_path.name}: GT={num_gt}, Pred={num_pred}")

    # --- 2. CALCULATE IMAGE-LEVEL METRICS ---
    
    precision_img = TP_img / (TP_img + FP_img) if (TP_img + FP_img) > 0 else 0.0
    recall_img = TP_img / (TP_img + FN_img) if (TP_img + FN_img) > 0 else 0.0
    f1_score_img = 2 * precision_img * recall_img / (precision_img + recall_img) if (precision_img + recall_img) > 0 else 0.0

    # --- 3. FINAL SUMMARY ---
    
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"Total images:              {len(image_files)}")
    
    print(f"\n### Image-Level Metrics (Detection/No-Detection)")
    print(f"  Description: 'Was *any* polyp found in an image that *had* one?'")
    print(f"  Images with GT & Pred (TP_img): {TP_img}")
    print(f"  Images with no GT & Pred (FP_img): {FP_img}")
    print(f"  Images with GT & no Pred (FN_img): {FN_img}")
    print(f"  Images with no GT & no Pred (TN_img): {TN_img}")
    print(f"  ---------------------------------")
    print(f"  Precision (Image-Level): {precision_img:.4f}")
    print(f"  Recall (Image-Level):    {recall_img:.4f}")
    print(f"  F1-score (Image-Level):  {f1_score_img:.4f}")

    print(f"\n### Polyp-Level Metrics (Object-by-Object)")
    print(f"  Description: 'Of all {total_gt_polyps} polyps, how many were found?'")
    print(f"  (Note: This re-runs validation using model.val() for robust metrics)")
    
    metrics = None
    if data_yaml_path:

        metrics = evaluate_model_seg(
            model_path=model_path, 
            data_yaml_path=data_yaml_path, 
            split_name=image_dir.name, 
            conf=conf_threshold, 
            iou=iou
        )
    else:
        print("\nNote: 'data_yaml_path' not provided. Cannot compute robust polyp-level metrics.")

    print(f"\nPredictions saved with GT overlay to: {save_dir.resolve()}")
    print(f"{'='*70}")
    
    # Return the metrics of the mask segmentation
    if metrics and hasattr(metrics, 'seg'):
        mask_p = metrics.seg.p[0] if len(metrics.seg.p) > 0 else 0.0
        mask_r = metrics.seg.r[0] if len(metrics.seg.r) > 0 else 0.0
        mask_f1 = 2 * mask_p * mask_r / (mask_p + mask_r) if (mask_p + mask_r) > 0 else 0.0
        return mask_p, mask_r, mask_f1, metrics.seg.map50, metrics.seg.map
    else:
        # Fallback if metrics couldn't be calculated
        return None, None, None, None, None
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ========== STEP 1: Dataset Preparation ==========
print("\n" + "="*70)
print("STEP 1: DATASET PREPARATION (SEGMENTATION)")
print("="*70)

converter = KvasirToYOLOSeg(IMAGE_DIR, MASK_DIR, JSON_PATH, OUTPUT_DIR, seed=SEED)
converter.prepare_dataset(train_split=0.7, val_split=0.2, test_split=0.1)



def find_best_hyperparameters():
    """Search for best_hyperparameters.yaml in tune directory."""
    tune_dir = Path("Kvasir-mask/tune")
    
    if not tune_dir.exists():
        return None
    
    # Search recursively for the file
    for yaml_file in tune_dir.rglob("best_hyperparameters.yaml"):
        print(f"Found hyperparameters at: {yaml_file}")
        return yaml_file
    
    return None


def train_yolo_seg_with_tuned_params(data_yaml_path, best_hyperparameters, 
                                      model_size=MODEL_SIZE, epochs=100, img_size=640, 
                                      batch_size=BATCH_SIZE, workers=4):
    """Train with tuned hyperparameters."""
    
    # Device detection
    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    model = YOLO(f'yolo11{model_size}-seg.pt')

    # ✅ Merge your fixed settings with tuned parameters
    training_args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'name': 'polyp_segmentation_v11_tuned',
        'patience': 10,
        'save': True,
        'device': device,
        'workers': workers,
        'optimizer': 'AdamW',
        'project': 'Kvasir-mask',
        
        # Fixed settings (always use these)
        'multi_scale': True,
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'augment': True,
        'auto_augment': 'randaugment',
        'mask_ratio': 4,
        'overlap_mask': True,
        
        # ✅ Add tuned hyperparameters (these override defaults)
        **best_hyperparameters  # This unpacks the dictionary
    }

    results = model.train(**training_args)
    return model
# ========== STEP 2: Train YOLOv11-seg ==========
print("\n" + "="*70)
print("STEP 2: TRAINING YOLOv11-seg MODEL")
print("="*70)

tuned_params_path = find_best_hyperparameters()

if tuned_params_path and tuned_params_path.exists():
    print(f"\n✓ Found tuned hyperparameters at: {tuned_params_path}")
    
    with open(tuned_params_path, 'r') as f:
        best_hyperparameters = yaml.safe_load(f)
    
    print("\n📋 Loaded Hyperparameters:")
    for key, value in best_hyperparameters.items():
        print(f"  {key}: {value}")
    
    # Train with tuned parameters
    model = train_yolo_seg_with_tuned_params(
        data_yaml_path=DATA_YAML,
        best_hyperparameters=best_hyperparameters,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    BEST_MODEL_PATH = Path("Kvasir-mask/polyp_segmentation_v11_tuned/weights/best.pt")
else:
    print("\n⚠️  No tuned parameters found, using defaults...")
    model = train_yolo_seg(
        data_yaml_path=DATA_YAML,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        lr0=1e-4  # Good starting point for nano
    )
    BEST_MODEL_PATH = Path("Kvasir-mask/polyp_segmentation_v11/weights/best.pt")

print(f"\n✓ Using model: {BEST_MODEL_PATH}")
#BEST_MODEL_PATH = Path("Kvasir-mask/polyp_segmentation_v11_tuned/weights/best.pt")

# ========== Full Test Set Evaluation ==========
print("\n" + "="*70)
print("STEP 4B: RUNNING INFERENCE ON ALL TEST IMAGES")
print("="*70)

predict_on_all_images_seg(
    BEST_MODEL_PATH, 
    f"{OUTPUT_DIR}/images/test",
    data_yaml_path=DATA_YAML,
    conf_threshold=0.001,
    iou=0.5,#increase to be more lax
    save_dir='test_predictions_seg'
)
print("\n" + "="*70)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
    