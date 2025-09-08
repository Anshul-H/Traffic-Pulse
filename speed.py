import cv2
import pandas as pd
import math
from ultralytics import YOLO
from tracker import Tracker
from datetime import datetime

def estimate_speed(p1, p2, fps, ppm=8):
    """Estimate speed in km/h from two points, frames per second, and pixels-per-meter scale."""
    distance_pixels = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    distance_meters = distance_pixels / ppm
    speed_mps = distance_meters * fps
    speed_kmph = speed_mps * 3.6
    return speed_kmph

def run_speed_estimation(video_path, output_path):
    model = YOLO('yolov8s.pt')

    with open("coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1020, 500))

    tracker = Tracker()
    count = 0
    log_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        timestamp = count / fps

        results = model.predict(frame)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")

        det_list = []
        for _, row in px.iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            class_id = int(row[5])
            label = class_list[class_id]
            if 'car' in label:
                det_list.append([x1, y1, x2, y2])

        bbox_id = tracker.update(det_list)

        for bbox in bbox_id:
            x3, y3, x4, y4, obj_id = bbox
            cx = (x3 + x4) // 2
            cy = (y3 + y4) // 2
            center = (cx, cy)

            # Estimate speed
            speed = 0
            history = tracker.track_history[obj_id]
            if len(history) >= 2:
                speed = estimate_speed(history[-2], history[-1], fps)
                speed = round(speed, 2)

            # Draw info
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {obj_id} | {speed} km/h", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Save to CSV log
            log_data.append({
                'timestamp_sec': round(timestamp, 2),
                'vehicle_id': obj_id,
                'x': cx,
                'y': cy,
                'speed_kmph': speed
            })

        out.write(frame)

    cap.release()
    out.release()

    # Save CSV log
    df = pd.DataFrame(log_data)
    df.to_csv('static/vehicle_log.csv', index=False)

    print(f"Processing complete. Output saved to {output_path}")
