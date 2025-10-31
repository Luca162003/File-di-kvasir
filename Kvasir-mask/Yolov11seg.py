import os
import sys
import json
import glob
import random
import shutil
from pathlib import Path
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from ultralytics import YOLO

print("Current working directory:", os.getcwd())
os.chdir("/users/lucatognari/Pitone/File-di-python/File-di-kvasir") #metti come directory il path del progetto, all'interno del quale si trova la cartella kvasir-mask


# ========== CONFIGURATION ==========
IMAGE_DIR = "Kvasir-mask/images"          # All 2500 images
MASK_DIR = "Kvasir-mask/masks"            # Masks for 2000 polyps
JSON_PATH = "Kvasir-mask/bounding-boxes.json"  # Bounding boxes
OUTPUT_DIR = "Kvasir-mask/kvasir_yolo_seg_dataset"
MODEL_SIZE = 'm'  # YOLOv11-nano
BATCH_SIZE = 16   # Increase for nano
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
        epsilon_multiplier = 0.000001

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
    
    # NOTE: _voc_to_yolo_bbox is included but NO LONGER CALLED in _process_split
    @staticmethod
    def _voc_to_yolo_bbox(bbox, img_width, img_height):
        """Convert Pascal VOC bbox to YOLO format with validation."""
        xmin = float(bbox['xmin'])
        ymin = float(bbox['ymin'])
        xmax = float(bbox['xmax'])
        ymax = float(bbox['ymax'])

        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        return [0, x_center, y_center, width, height]

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

        # Split each category proportionally
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

        rnd = random.Random(self.seed)
        for split_imgs in splits.values():
            rnd.shuffle(split_imgs)

        print(f"\n{'='*70}")
        print("DATASET SPLIT")
        # ... (split summary printing remains unchanged) ...
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

        print(f"\n✓ Segmentation dataset created!")
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
                    label_path.touch()
                    continue

                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Warning: Could not read mask for {img_id}")
                    label_path.touch()
                    continue

                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                # Get the SINGLE clean polygon (from the mask_to_polygon logic)
                polygons = self.mask_to_polygon(mask)

                with open(label_path, 'w') as f:
                    # Write segmentation polygon(s)
                    if polygons:
                        for polygon in polygons:
                            norm_poly = self.normalize_polygon(polygon, w, h)
                            # Format for YOLO: class_id x1 y1 x2 y2 ...
                            coords = ' '.join(f"{x:.6f} {y:.6f}" for x, y in norm_poly)
                            f.write(f"0 {coords}\n")
                    else:
                        print(f"Warning: No valid polygons for {img_id}")

                # !!! CRITICAL FIX: The section that wrote the original BBOX line (0 x y w h) 
                # has been removed to prevent the redundant second bounding box in the label file.

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
        patience=10,
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
def evaluate_model_seg(model_path, data_yaml_path, split='val', conf = 0.001, iou = 0.5):
    """Evaluate segmentation model."""
    model = YOLO(model_path)
    
    print(f"Running validation with NMS IOU = {iou} and Conf = {conf}")
    metrics = model.val(
        data=data_yaml_path, 
        split=split,
        conf=conf,
        iou=iou
    )

    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS ON {split.upper()} SET")
    print(f"{'='*70}")
    
    # Box metrics
    print(f"Box mAP@0.50     : {metrics.box.map50:.4f}")
    print(f"Box mAP@0.50-95  : {metrics.box.map:.4f}")
    
    # Mask metrics (segmentation)
    print(f"\nMask mAP@0.50    : {metrics.seg.map50:.4f}")
    print(f"Mask mAP@0.50-95 : {metrics.seg.map:.4f}")
    
    if hasattr(metrics.box, 'p') and len(metrics.box.p) > 0:
        print(f"\nBox Precision    : {metrics.box.p[0]:.4f}")
    if hasattr(metrics.box, 'r') and len(metrics.box.r) > 0:
        print(f"Box Recall       : {metrics.box.r[0]:.4f}")
    
    if hasattr(metrics.seg, 'p') and len(metrics.seg.p) > 0:
        print(f"\nMask Precision   : {metrics.seg.p[0]:.4f}")
    if hasattr(metrics.seg, 'r') and len(metrics.seg.r) > 0:
        print(f"Mask Recall      : {metrics.seg.r[0]:.4f}")
    
    print(f"{'='*70}\n")

    return metrics


def predict_and_visualize_seg(model_path, image_path, conf_threshold=0.25, iou = 0.5):
    """Run inference and visualize segmentation results on a single image."""
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold, iou=iou)

    for result in results:
        # Plot with both boxes and masks
        img_with_results = result.plot()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_with_results, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Polyp Segmentation (confidence > {conf_threshold})', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Print detections
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            print(f"\n✓ Detected {len(result.boxes)} polyp(s):")
            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0].item())
                print(f"  Polyp {i+1}: confidence = {conf:.3f}")
        else:
            print("\n✓ No polyps detected (healthy image)")


def predict_on_all_images_seg(model_path, image_dir, data_yaml_path=None, 
                               conf_threshold=0.25, save_dir='predictions_seg', iou = 0.5):
    """
    Run inference on ALL images in a directory, save results, and compute statistics.
    Works with segmentation models - saves images with boxes + masks.
    
    Args:
        model_path: Path to trained segmentation model
        image_dir: Directory with test or val images
        data_yaml_path: Optional path to data.yaml for mAP evaluation
        conf_threshold: Confidence threshold
        save_dir: Directory to save prediction images
    """
    model = YOLO(model_path)
    image_dir = Path(image_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    label_dir = image_dir.parent.parent / 'labels' / image_dir.name
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"\n{'='*70}")
    print(f"RUNNING INFERENCE ON ALL {len(image_files)} IMAGES")
    print(f"{'='*70}\n")

    total_gt_polyps = 0
    total_pred_polyps = 0
    images_with_polyps = 0

    TP = FP = FN = TN = 0

    for img_path in image_files:
        # Check ground truth
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                gt_boxes = [line for line in f if line.strip()]
                num_gt = len(gt_boxes)
        else:
            num_gt = 0

        # Run inference
        results = model(str(img_path), conf=conf_threshold, iou=iou, verbose=False)

        for result in results:
            # Save image with boxes AND masks
            img_with_results = result.plot()
            output_path = save_dir / f"pred_{img_path.name}"
            img_to_save = np.ascontiguousarray(img_with_results, dtype=np.uint8)
            cv2.imwrite(str(output_path), img_with_results)

            # Count predictions
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                num_pred = len(result.boxes)
                total_pred_polyps += num_pred
                images_with_polyps += 1
                print(f"✓ {img_path.name}: {num_pred} polyp(s)")
            else:
                num_pred = 0
                print(f"  {img_path.name}: No polyps detected")

            # Contingency matrix (image-level)
            if num_gt > 0 and num_pred > 0:
                TP += 1
            elif num_gt == 0 and num_pred > 0:
                FP += 1
            elif num_gt > 0 and num_pred == 0:
                FN += 1
            elif num_gt == 0 and num_pred == 0:
                TN += 1

            total_gt_polyps += num_gt

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # mAP evaluation (uses YOLO's built-in validation)
    map50_box = map5095_box = None
    map50_mask = map5095_mask = None
    
    if data_yaml_path:
        try:
            metrics = model.val(
                data=data_yaml_path, 
                split=image_dir.name, 
                conf=conf_threshold, 
                iou=iou,
                verbose=False
            )
            
            # Box metrics
            map50_box = float(getattr(metrics.box, 'map50', 0.0))
            map5095_box = float(getattr(metrics.box, 'map', 0.0))
            
            # Mask metrics (segmentation specific)
            if hasattr(metrics, 'seg'):
                map50_mask = float(getattr(metrics.seg, 'map50', 0.0))
                map5095_mask = float(getattr(metrics.seg, 'map', 0.0))
        except Exception as e:
            print(f"\nWarning: Could not compute mAP metrics: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"INFERENCE SUMMARY")
    print(f"{'='*70}")
    print(f"Total images:              {len(image_files)}")
    print(f"Images with polyps:        {images_with_polyps}")
    print(f"Images without polyps:     {len(image_files) - images_with_polyps}")
    print(f"Total ground truth polyps: {total_gt_polyps}")
    print(f"Total predicted polyps:    {total_pred_polyps}")
    
    print(f"\nContingency Matrix (image-level):")
    print(f"  TP (GT & Pred):           {TP}")
    print(f"  FP (No GT, Pred):         {FP}")
    print(f"  FN (GT, No Pred):         {FN}")
    print(f"  TN (No GT, No Pred):      {TN}")
    
    print(f"\nImage-Level Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1_score:.4f}")
    
    if map50_box is not None:
        print(f"\nBox Detection Metrics:")
        print(f"  mAP@0.50:     {map50_box:.4f}")
        print(f"  mAP@0.50-0.95: {map5095_box:.4f}")
    
    if map50_mask is not None:
        print(f"\nMask Segmentation Metrics:")
        print(f"  mAP@0.50:     {map50_mask:.4f}")
        print(f"  mAP@0.50-0.95: {map5095_mask:.4f}")
    
    print(f"\nPredictions saved to: {save_dir.resolve()}")
    print(f"{'='*70}")
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

model = YOLO(f'yolo11{MODEL_SIZE}-seg.pt')  # Replace with your model path if different

search_space = {
    "lr0": (1e-5, 1e-3),
    "lrf": (0.01, 0.1),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.001),
    "dfl": (1.0, 2.0),
    "hsv_h": (0.0, 0.02),
    "hsv_s": (0.0, 0.3),
    "hsv_v": (0.0, 0.3),
    "degrees": (0.0, 10.0),
    "translate": (0.0, 0.1),
    "scale": (0.0, 0.2),
    "shear": (0.0, 2.0),
    "perspective": (0.0, 0.0001),
    "flipud": (0.0, 1.0),
    "fliplr": (0.0, 1.0),
    "box": (3.0, 7.5),     # Box loss weight
    "cls": (0.2, 2.0) 
}

model.tune(
    data="Kvasir-mask/kvasir_yolo_seg_dataset/data.yaml",
    epochs=10,
    iterations=100,
    optimizer="AdamW",
    space=search_space,
    plots=True,
    save=True,
    val=True,
    project="Kvasir-mask/tune"  # ✅ Saves results in ./Kvasir_mask/tune/
)
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

# ========== STEP 3: Evaluate Model ==========
print("\n" + "="*70)
print("STEP 3: MODEL EVALUATION")
print("="*70)

evaluate_model_seg(BEST_MODEL_PATH, DATA_YAML, split='test', conf=0.001, iou=0.5) 
# ========== STEP 4A: Quick Visual Test ==========
print("\n" + "="*70)
print("STEP 4A: QUICK VISUAL TEST (Single Image)")
print("="*70)

test_images = list(Path(f"{OUTPUT_DIR}/images/test").glob("*.*"))
if test_images:
    polyp_img = random.choice([img for img in test_images[:10]])
    print(f"\nTesting on: {polyp_img.name}")
    predict_and_visualize_seg(BEST_MODEL_PATH, str(polyp_img), conf_threshold=0.25) 

# ========== STEP 4B: Full Test Set Evaluation ==========
print("\n" + "="*70)
print("STEP 4B: RUNNING INFERENCE ON ALL TEST IMAGES")
print("="*70)

predict_on_all_images_seg(
    BEST_MODEL_PATH, 
    f"{OUTPUT_DIR}/images/test",
    data_yaml_path=DATA_YAML,
    conf_threshold=0.25,
    iou=0.5,#increase to be more lax
    save_dir='test_predictions_seg'
)
print("\n" + "="*70)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
    