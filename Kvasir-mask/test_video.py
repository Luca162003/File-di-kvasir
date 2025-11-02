from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
import shutil

def is_video_readable(video_path):
    cap = cv2.VideoCapture(str(video_path))
    readable = cap.isOpened()
    cap.release()
    return readable

def predict_on_video(model_path, video_path, conf_threshold=0.5, output_filename='output_video.mp4'):
    print(f"\n{'='*70}")
    print(f"RUNNING INFERENCE ON VIDEO")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Confidence: {conf_threshold}")
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ ERROR: Video file not found at {video_path}")
        return
    if not is_video_readable(video_path):
        print(f"❌ ERROR: Video file cannot be opened. Check format/codec.")
        return

    model = YOLO(model_path)
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

    project_dir = '/home/luca/Desktop/Luca/File-di-kvasir_Daniele'
    run_name = 'polyp_detection'

    results = model.predict(
        source=str(video_path),
        conf=conf_threshold,
        iou=0.5,
        device=device,
        save=True,
        show=False,
        stream=True,
        verbose=True,
        project=project_dir,
        name=run_name
    )

    # Rename output video
    output_dir = Path(project_dir) / run_name / 'predict'
    default_video = output_dir / 'video.mp4'
    custom_video = output_dir / output_filename

    if default_video.exists():
        shutil.move(str(default_video), str(custom_video))
        print(f"✅ Output video renamed to: {custom_video}")
    else:
        print("⚠️ Output video not found for renaming.")

    # Summary
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
    if total_frames > 0:
        detection_rate = frames_with_polyps / total_frames * 100
        print(f"Detection rate: {detection_rate:.1f}%")
    else:
        print("⚠️ No frames were processed. Check video format or decoding errors.")
    print(f"\n📹 Final video saved to: {custom_video}")
    print(f"{'='*70}")

video_paths=['/home/luca/Desktop/Luca/File-di-kvasir_Daniele/Exp3_squared/2021.06.15 M9_W6_squared.mp4',
                '/home/luca/Desktop/Luca/File-di-kvasir_Daniele/Exp3_squared/2021.05.06 M9_D0_squared.mp4',
                '/home/luca/Desktop/Luca/File-di-kvasir_Daniele/Exp3_squared/2021.05.31 M9_W4_squared.mp4',
                '/home/luca/Desktop/Luca/File-di-kvasir_Daniele/Exp3_squared/2021.07.01 M9_W9_squared.mp4']
for path in video_paths:
    predict_on_video(
        model_path='/home/luca/Desktop/Luca/File-di-kvasir_Daniele/Kvasir-mask/polyp_segmentation_v11_tuned/weights/best.pt',
        video_path=path,
        conf_threshold=0.2
    )
