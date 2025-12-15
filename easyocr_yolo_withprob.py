import os
import sys
import argparse
import glob
import time
import re

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source: file, folder, video, usb0, picamera0', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.7)
parser.add_argument('--resolution', help='Resolution in WxH (example: "640x480")', default=None)
parser.add_argument('--record', help='Record results to "demo1.avi"', action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh) # Ensure this is a float
user_res = args.resolution
record = args.record
ocr_prob = 0.1

# Initialize EasyOCR Reader (loads model once into memory)
# 'en' is usually sufficient for number plates. Add other languages if needed.
print("Initializing OCR engine...")
reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have a compatible GPU, standard Pi5 does not.

# Create directory for saved plates if it doesn't exist
save_dir = 'detected_plates'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(0)

# Load the YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input source type
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list: source_type = 'image'
    elif ext in vid_ext_list: source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid.')
    sys.exit(0)

# Parse resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video/camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify --resolution to record.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW,resH))

# Initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Tableau 10 color scheme
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

print("Starting inference...")
while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('Finished processing images.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret: break
    elif source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None: break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None: break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    height, width, _ = frame.shape

    for i in range(len(detections)):
        conf = detections[i].conf.item()
        if conf > min_thresh:
            # Get coordinates
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy

            # Ensure coordinates are within frame bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width, xmax)
            ymax = min(height, ymax)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # --- OCR PROCESSING START ---
            # Crop the detected plate
            plate_crop = frame[ymin:ymax, xmin:xmax]

            ocr_text = ""
            # Only run OCR if the crop has a valid size
            if plate_crop.size > 0:
                # crop_img = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                # crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                # crop_gray = cv2.GaussianBlur(crop_gray, (1,1), 10)
                # structuring_element = np.zeros((40, 40), np.uint8)
                # structuring_element[1:-1, 1:-1] = 1
                # final_img = cv2.morphologyEx(crop_gray, cv2.MORPH_BLACKHAT, structuring_element)
                try:
                    # detail=0 returns simple list of detected text strings
                    ocr_results = reader.readtext(plate_crop, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    filtered_text = []
                    for (bbox, text, prob) in ocr_results:
                        if prob >= ocr_prob:
                            filtered_text.append(text)
                            ocr_prob = prob
                    ocr_text = "".join(filtered_text).strip()
                except Exception as e:
                     print(f"OCR Failed: {e}")

            # Prepare label with OCR text
            label = f'{classname}: {int(conf*100)}%'
            if ocr_text:
                label += f' | {ocr_text}'
                
                # Save cropped image renamed with detected text
                # Use regex to keep only alphanumeric for safe filename
                if ocr_text and len(ocr_text) >= 6:
                    safe_text = re.sub(r'[^a-zA-Z0-9]', '', ocr_text)
                    if safe_text:
                        img_name = os.path.join(save_dir, f'{safe_text}_{ocr_prob:0.2f}.jpg')
                        # Optional: Add timestamp to filename if you expect duplicates: 
                        # img_name = os.path.join(save_dir, f'{safe_text}_{int(time.time())}.jpg')
                        cv2.imwrite(img_name, plate_crop)
            # --- OCR PROCESSING END ---

            # Draw bounding box and label
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            object_count += 1

    # Draw status info
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    cv2.imshow('YOLO+OCR Results', frame)
    if record: recorder.write(frame)

    # Key controls
    wait_time = 0 if source_type in ['image', 'folder'] else 1
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'): cv2.waitKey(0)
    elif key == ord('p'): cv2.imwrite('capture.png', frame)

    # FPS calculation
    t_stop = time.perf_counter()
    frame_rate_calc = 1/(t_stop - t_start)
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len: frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
if 'cap' in locals():
    if source_type == 'picamera': cap.stop()
    else: cap.release()
if record: recorder.release()
cv2.destroyAllWindows()