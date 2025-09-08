# streamlit_app.py

import streamlit as st
from track import run_tracker

video_path = st.text_input("Enter path to your video or leave empty for webcam")

if st.button("Run Vehicle Tracker"):
    if video_path:
        run_tracker(video_path)
    else:
        run_tracker()  # default webcam
