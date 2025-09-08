import os
import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
from datetime import datetime
import csv
from collections import defaultdict
import uuid
from huggingface_hub import hf_hub_download

# Class ID to label mapping
CLASS_MAP = {
    2: 'car',
    3: 'motorbike',
    5: 'bus',
    7: 'truck'
}

# Function to download YOLOv8 weights if not present
def get_model_path():
    model_file = "yolov8l.pt"
    if not os.path.exists(model_file):
        # Download from the existing Hugging Face repo
        model_file = hf_hub_download(
            repo_id="lkk688/yolov8l-model",
            filename="yolov8l.pt"
        )
    return model_file

@st.cache_resource
def load_model():
    model_path = get_model_path()
    return YOLO(model_path)

def run_tracker(video_path=0):
    model = load_model()  # Load model safely
    vehicle_counts = defaultdict(int)

    st.title("🚗 Real-Time Vehicle Counter")
    stframe = st.empty()
    chart_placeholder = st.empty()

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = CLASS_MAP.get(cls_id)
                if label:
                    vehicle_counts[label] += 1
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
                    cv2.putText(frame, label, tuple(xyxy[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        stframe.image(frame, channels='BGR', use_column_width=True)

        df = pd.DataFrame(vehicle_counts.items(), columns=['Vehicle', 'Count'])
        fig = px.bar(df, x='Vehicle', y='Count', title='Live Vehicle Count')
        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{uuid.uuid4()}")

        if sum(vehicle_counts.values()) % 50 == 0:
            os.makedirs("static", exist_ok=True)
            with open("static/vehicle_log.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now()] + list(vehicle_counts.values()))

    cap.release()
