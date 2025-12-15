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
parser.add_argument('--thresh', help='Minimum confidence threshold for Detection', default=0.4)
parser.add_argument('--ocr_thresh', help='Minimum confidence to trigger OCR (should be higher than detection thresh)', default=0.6)
parser.add_argument('--resolution', help='Resolution in WxH (example: "640x480")', default=None)
parser.add_argument('--record', help='Record results to "demo1.avi"', action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
ocr_thresh = float(args.ocr_thresh)
user_res = args.resolution
record = args.record

# --- OCR CONFIGURATION ---
# Cooldown in seconds to prevent continuous OCR on the same blurry stationary car.
# Adjust this: Increase if you get too many duplicate reads, decrease if you miss fast cars.
OCR_COOLDOWN = 3.0 
last_ocr_time = 0
last_detected_text = "Waiting for plate..."

print("Initializing OCR engine...")
reader = easyocr.Reader(['en'], gpu=False)

save_dir = 'detected_plates'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

# Input source setup
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list: source_type = 'image'
    elif ext in vid_ext_list: source_type = 'video'
    else: sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

if record:
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW,resH))

# Initialize sources
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

print("Starting inference...")
while True:
    t_start = time.perf_counter()

    # Frame loading logic
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list): break
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

    if resize: frame = cv2.resize(frame, (resW, resH))
    height, width, _ = frame.shape

    # 1. Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Variables to find the BEST candidate for OCR in this frame
    best_conf = 0
    best_crop = None
    best_coords = None
    
    current_time = time.time()

    # 2. Loop through all detections to draw boxes AND find best candidate
    for i in range(len(detections)):
        conf = detections[i].conf.item()
        if conf > min_thresh:
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(width, xmax), min(height, ymax)
            
            classidx = int(detections[i].cls.item())
            
            # Draw standard detection box
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            # Check if this is the best candidate so far for OCR
            if conf > best_conf:
                best_conf = conf
                best_coords = (xmin, ymin, xmax, ymax)
                best_crop = frame[ymin:ymax, xmin:xmax]

    # 3. OCR TRIGGER LOGIC
    # Only run OCR if:
    # A) We found a plate with high confidence (> ocr_thresh)
    # B) Enough time has passed since the last read (> OCR_COOLDOWN)
    if best_conf > ocr_thresh and (current_time - last_ocr_time) > OCR_COOLDOWN:
        if best_crop is not None and best_crop.size > 0:
            print(f"Triggering OCR! Conf: {best_conf:.2f}")
            
            # OPTIONAL: Flash the screen or draw a special box to show a capture happened
            cv2.rectangle(frame, (best_coords[0], best_coords[1]), (best_coords[2], best_coords[3]), (0,255,0), 5)
            
            try:
                # Run OCR on the frozen crop
                ocr_results = reader.readtext(best_crop, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                text_clean = "".join(ocr_results).strip()
                
                if text_clean:
                    last_detected_text = text_clean # Update persistent display
                    last_ocr_time = current_time    # Reset cooldown
                    
                    # Save the clear still
                    safe_text = re.sub(r'[^a-zA-Z0-9]', '', text_clean)
                    img_name = os.path.join(save_dir, f'{safe_text}.jpg')
                    cv2.imwrite(img_name, best_crop)
                    print(f"OCR Success: {text_clean}")
            except Exception as e:
                print(f"OCR Failed: {e}")

    # 4. Draw Persistent Status Info on Screen
    # Main status box at top of screen
    cv2.rectangle(frame, (0, 0), (width, 40), (0,0,0), -1) # Black banner background
    
    # Show FPS
    cv2.putText(frame, f'FPS: {avg_frame_rate:0.1f}', (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Show Last Read Plate persistently
    # Green text if recently read, White if old
    txt_color = (0, 255, 0) if (current_time - last_ocr_time) < OCR_COOLDOWN else (200, 200, 200)
    cv2.putText(frame, f'LAST READ: {last_detected_text}', (160, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_color, 2)

    # Display and control
    # If you are on headless Pi, comment out the next 3 lines
    cv2.imshow('License Plate Reader', frame)
    if record: recorder.write(frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

    # FPS calc
    t_stop = time.perf_counter()
    frame_rate_buffer.append(1/(t_stop - t_start))
    if len(frame_rate_buffer) > fps_avg_len: frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
if 'cap' in locals():
    if source_type == 'picamera': cap.stop()
    else: cap.release()
if record: recorder.release()
cv2.destroyAllWindows()