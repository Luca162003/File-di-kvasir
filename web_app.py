import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
# Import your model loading logic here
# from my_model import load_efficientnet, predict

st.title("🚀 EfficientNet Video Classifier")

# 1. Sidebar for Model Settings
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# 2. File Uploader
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
   
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    vf = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    fps = vf.get(cv2.CAP_PROP_FPS)
    width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    st.info(f"Processing video: {width}x{height} at {fps} FPS")
    
    progress_bar = st.progress(0)
    frame_count = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_window = st.image([])

    count = 0
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)
        
        count += 1
        progress_bar.progress(count / frame_count)

    vf.release()
    st.success("Processing Complete!")