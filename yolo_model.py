import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Image source, can be image file, image folder, video file, USB camera, or IP camera URL', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH format', default=None)
parser.add_argument('--record', help='Record results and save as "demo1.avi". Must specify --resolution.', action='store_true')
args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Determine source type
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(0)
elif 'http' in img_source:  # IP camera support
    source_type = 'ip'
    cap = cv2.VideoCapture(img_source)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
    cap = cv2.VideoCapture(usb_idx)
else:
    print(f'Invalid source: {img_source}')
    sys.exit(0)

# Parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Setup recording
if record and source_type in ['video', 'usb', 'ip']:
    if not user_res:
        print('Specify resolution for recording.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# Bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Inference loop
while True:
    t_start = time.perf_counter()
    
    if source_type in ['video', 'usb', 'ip']:
        ret, frame = cap.read()
        if not ret:
            print('Video stream ended or cannot read frames.')
            break
    
    if resize:
        frame = cv2.resize(frame, (resW, resH))
    
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0
    
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()
        
        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            cv2.putText(frame, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            object_count += 1
    
    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow('YOLO Detection', frame)
    if record:
        recorder.write(frame)
    
    key = cv2.waitKey(5)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)
    
    t_stop = time.perf_counter()
    fps = 1 / (t_stop - t_start)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

# Cleanup
print('Closing program.')
if source_type in ['video', 'usb', 'ip']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
