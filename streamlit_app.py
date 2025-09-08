import sys
import os
import importlib.util

# Ensure Python can find track.py in the same folder
sys.path.append(os.path.dirname(__file__))

import streamlit as st
# Try importing run_tracker from track.py in the same directory
try:
    from track import run_tracker
except ModuleNotFoundError:
    st.error("track.py not found in the current directory. Please ensure track.py exists.")
    sys.exit(1)

st.set_page_config(page_title="🚗 Real-Time Vehicle Counter", layout="wide")
st.title("🚗 Real-Time Vehicle Counter")

st.markdown("""
This app uses YOLOv8 to detect and count vehicles in real-time from your webcam or a video file.
""")

# Sidebar for video input
video_source = st.sidebar.radio(
    "Select Video Source:",
    ("Webcam", "Upload Video")
)

video_path = 0  # default to webcam
if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        video_path = "temp_video.mp4"

# Start tracking
if st.button("Start Tracking"):
    run_tracker(video_path)
