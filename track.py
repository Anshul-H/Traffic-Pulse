import cv2
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
from datetime import datetime
import csv
from collections import defaultdict
import uuid  # For generating unique keys
import os

# Class ID to label mapping
CLASS_MAP = {
    2: 'car',
    3: 'motorbike',
    5: 'bus',
    7: 'truck'
}

# Cache the model to prevent reloading
@st.cache_resource
def load_model():
    """
    Load YOLOv8 model safely.
    Automatically downloads official weights if yolov8l.pt is missing.
    """
    return YOLO("yolov8l.pt")


def run_tracker(video_path=0):  # 0 for webcam
    """
    Run real-time vehicle tracking and counting in Streamlit.
    :param video_path: 0 for webcam, or path to video file
    """
    model = load_model()  # Load model safely
    vehicle_counts = defaultdict(int)

    stframe = st.empty()
    chart_placeholder = st.empty()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Cannot open video source. Please check your webcam or video file.")
        return

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
                    cv2.putText(frame, label, tuple(xyxy[:2]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame in Streamlit
        stframe.image(frame, channels='BGR', use_container_width=True)

        # Display live vehicle count chart
        df = pd.DataFrame(vehicle_counts.items(), columns=['Vehicle', 'Count'])
        fig = px.bar(df, x='Vehicle', y='Count', title='Live Vehicle Count')
        chart_placeholder.plotly_chart(fig, use_container_width=True, key=str(uuid.uuid4()))

        # Optional: save to CSV every 50 vehicles
        if sum(vehicle_counts.values()) % 50 == 0 and sum(vehicle_counts.values()) != 0:
            os.makedirs("static", exist_ok=True)
            with open("static/vehicle_log.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now()] + list(vehicle_counts.values()))

    cap.release()
