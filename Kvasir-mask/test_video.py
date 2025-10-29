from pathlib import Path
from ultralytics import YOLO
import torch

def predict_on_video(model_path, video_path, conf_threshold=0.5,
                     save_path='output_video.mp4'):
    """
    Run polyp segmentation on a video and save results.
    
    Args:
        model_path: Path to your trained model
        video_path: Path to input video
        conf_threshold: Confidence threshold
        save_path: Where to save output video
    """
    
    print(f"\n{'='*70}")
    print(f"RUNNING INFERENCE ON VIDEO")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Confidence: {conf_threshold}")
    
    # Load model
    model = YOLO(model_path)
    
    # ✅ Device detection and setup
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"🚀 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = 'cpu'
        print(f"⚠️  Using CPU (slower)")
    
    # Move model to device
    model.to(device)
    
    # Run inference on video
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        iou=0.5,
        device=device,
        save=True,              # ✅ CHANGED: Save output video
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
    
    print(f"\n{'='*70}")
    print(f"VIDEO PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total frames: {total_frames}")
    print(f"Frames with polyps detected: {frames_with_polyps}")
    print(f"Detection rate: {frames_with_polyps/total_frames*100:.1f}%")
    print(f"\n📹 Output video saved to: video_results/polyp_detection/")
    print(f"{'='*70}")

# ✅ ALSO FIX THIS: Second parameter should be VIDEO PATH, not folder!
predict_on_video(
    model_path='/Users/lucatognari/Pitone/File-di-python/File-di-kvasir/Kvasir-mask/polyp_segmentation_v11/weights/best.pt',
    video_path='/Users/lucatognari/Downloads/qEndoscopic Stomach Polyp Removal [Tz2ktVJVWcI].mp4',  # ✅ Put actual video file here!
    conf_threshold=0.5
)