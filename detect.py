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
parser.add_argument('--source', help='Image source, can be an image file ("test.jpg"), folder ("test_dir"), video file ("testvid.mp4"), USB camera index ("usb0"), or IP camera URL ("http://192.168.1.100:8080/video")', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")', default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH (example: "640x480"), otherwise match source resolution', default=None)
parser.add_argument('--record', help='Record results from video or webcam and save as "demo1.avi". Requires --resolution.', action='store_true')

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

# Load the model
model = YOLO(model_path, task='detect')
labels = model.names

# Determine source type
img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension {ext}.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith("http"):  # IP Camera
    source_type = 'ip'
else:
    print(f'Invalid source {img_source}.')
    sys.exit(0)

# Parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Check recording validity
if record:
    if source_type not in ['video', 'usb', 'ip']:
        print('Recording only works for video, USB camera, or IP camera.')
        sys.exit(0)
    if not user_res:
        print('Specify resolution for recording.')
        sys.exit(0)
    
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Initialize source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [file for file in glob.glob(img_source + '/*') if os.path.splitext(file)[1] in img_ext_list]
elif source_type in ['video', 'usb', 'ip']:
    cap_arg = img_source if source_type in ['video', 'ip'] else usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set resolution if specified
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

# Bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize tracking
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Start inference loop
while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed.')
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type in ['video', 'usb', 'ip']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Stream ended or unavailable.')
            break

    # Resize frame if required
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    # Process detections
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(detections[i].cls.item())
        conf = detections[i].conf.item()

        if conf > min_thresh:
            xmin, ymin, xmax, ymax = xyxy
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{labels[classidx]}: {int(conf * 100)}%'
            labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count += 1

    # Display FPS
    if source_type in ['video', 'usb', 'ip']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    # Display object count
    cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection', frame)

    # Record video if enabled
    if record:
        recorder.write(frame)

    # Handle user input
    key = cv2.waitKey(5 if source_type in ['video', 'usb', 'ip'] else 0)
    if key in [ord('q'), ord('Q')]: break
    elif key in [ord('s'), ord('S')]: cv2.waitKey()
    elif key in [ord('p'), ord('P')]: cv2.imwrite('capture.png', frame)

    # Calculate FPS
    frame_rate_calc = 1 / (time.perf_counter() - t_start)
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len: frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f'Average FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb', 'ip']: cap.release()
if record: recorder.release()
cv2.destroyAllWindows()
