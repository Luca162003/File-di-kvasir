from pathlib import Path
from ultralytics import YOLO
import torch
import cv2

def is_video_readable(video_path):
    """Check if video file can be opened."""
    cap = cv2.VideoCapture(str(video_path))
    readable = cap.isOpened()
    cap.release()
    return readable

def predict_on_video(model_path, video_path, conf_threshold=0.5, save_path='output_video.mp4'):
    """Run polyp detection on video file."""
    print(f"\n{'='*70}")
    print(f"RUNNING INFERENCE ON VIDEO")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Confidence: {conf_threshold}")
    
    video_path = Path(video_path)
    
    # Validate video file
    if not video_path.exists():
        print(f"❌ ERROR: Video file not found at {video_path}")
        return
    
    if not is_video_readable(video_path):
        print(f"❌ ERROR: Video file cannot be opened. Check format/codec.")
        return
    
    # Load model
    model = YOLO(model_path)
    
    # Device detection
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"🚀 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = 'cpu'
        print(f"⚠️  Using CPU (slower)")
    
    model.to(device)
    
    # Run prediction
    results = model.predict(
        source=str(video_path),
        conf=conf_threshold,
        iou=0.5,
        device=device,
        save=True,
        show=False,
        stream=True,
        verbose=True,
        project='video_results',
        name='polyp_detection'
    )
    
    # Process results
    total_frames = 0
    frames_with_polyps = 0
    
    for result in results:
        total_frames += 1
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            frames_with_polyps += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"VIDEO PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total frames: {total_frames}")
    print(f"Frames with polyps detected: {frames_with_polyps}")
    
    if total_frames > 0:
        detection_rate = frames_with_polyps / total_frames * 100
        print(f"Detection rate: {detection_rate:.1f}%")
    else:
        print("⚠️ No frames were processed. Check video format or decoding errors.")
    
    print(f"\n📹 Output video saved to: video_results/polyp_detection/")
    print(f"{'='*70}")

# Example usage
predict_on_video(
    model_path='/home/luca/Desktop/Luca/File-di-kvasir_Daniele/Kvasir-mask/polyp_segmentation_v11_tuned/weights/best.pt',
    video_path='/home/luca/hyper-kvasir-videos/videos/f611197c-acb7-4703-90aa-6a4a623c614f.avi',
    save_path='output_video.mp4',
    conf_threshold=0.5
)
