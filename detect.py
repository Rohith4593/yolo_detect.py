import os
import sys
import argparse
import time
import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO
from collections import deque

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file')
parser.add_argument('--source', required=True, help='Video or IP camera source')
parser.add_argument('--thresh', default=0.5, type=float, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='Resolution WxH')
parser.add_argument('--record', action='store_true', help='Record video output')
args = parser.parse_args()

# Load YOLO model
model = YOLO(args.model, task='detect')
labels = model.names

# Initialize video source
cap = cv2.VideoCapture(args.source)
if not cap.isOpened():
    print("Error: Unable to access video source.")
    sys.exit(0)

# Set resolution if specified
if args.resolution:
    resW, resH = map(int, args.resolution.split('x'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
else:
    resW, resH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Mouse control setup
screenW, screenH = pyautogui.size()
tracker = deque(maxlen=5)  # Store last 5 object positions for smoothing

# Performance tweaks for IP cameras
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired FPS if possible

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or unavailable.")
        break
    
    results = model(frame, verbose=False)
    detections = results[0].boxes
    
    max_conf = 0
    best_box = None
    for det in detections:
        conf = det.conf.item()
        if conf > max_conf and conf > args.thresh:
            max_conf = conf
            best_box = det.xyxy.cpu().numpy().squeeze().astype(int)

    if best_box is not None:
        xmin, ymin, xmax, ymax = best_box
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        tracker.append((cx, cy))
        
        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f'Pen: {int(max_conf * 100)}%', (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Move mouse cursor smoothly
        avg_x = int(np.mean([pt[0] for pt in tracker]))
        avg_y = int(np.mean([pt[1] for pt in tracker]))
        mouse_x = int((avg_x / resW) * screenW)
        mouse_y = int((avg_y / resH) * screenH)
        pyautogui.moveTo(mouse_x, mouse_y, duration=0.05)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
