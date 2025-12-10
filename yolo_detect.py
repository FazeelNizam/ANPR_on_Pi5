import sys
# This allows the virtual environment to see the system-installed Picamera2/Libcamera
sys.path.append('/usr/lib/python3/dist-packages')
# --------------------------

import os
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
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.7)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
try:
    min_thresh = float(args.thresh)
except ValueError:
    print("Invalid threshold value. Using default 0.7")
    min_thresh = 0.7
user_res = args.resolution
record = args.record

# --- OCR Initialization ---
print("Initializing EasyOCR... (This may take a moment)")
# Set gpu=True for Windows PC. It will use CPU if no compatible GPU is found.
reader = easyocr.Reader(['en'], gpu=False) 

# Create directory for saved plates
save_dir = 'detected_plates'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# -------------------------

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
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
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    # Check if picamera2 is available
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("Error: 'picamera2' module not found. Ensure you have installed python3-picamera2 via apt.")
        sys.exit(1)
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)
    
    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    cap = Picamera2()
    # Configure Picamera
    config = cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}) if resize else cap.create_video_configuration(main={"format": 'RGB888'})
    cap.configure(config)
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

print("Starting inference loop...")

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # --- LOAD FRAME LOGIC ---
    frame = None

    if source_type == 'image' or source_type == 'folder': 
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': 
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': 
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. Exiting program.')
            break

    elif source_type == 'picamera': 
        frame = cap.capture_array()
        if frame is not None:
             # Picamera returns RGB, OpenCV expects BGR
             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if frame is None:
        print('Error: Failed to capture frame. Skipping...')
        continue

    # Initialize frame dimensions
    frame_h, frame_w, _ = frame.shape
    # ----------------------------------------------------

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame, (resW, resH))
        # Update dimensions after resize
        frame_h, frame_w, _ = frame.shape

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Go through each detection
    for i in range(len(detections)):
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:
            
            # Get bounding box coordinates
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            
            # Ensure coordinates are within frame bounds (Uses fixed frame_w/frame_h)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(frame_w, xmax)
            ymax = min(frame_h, ymax)

            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # --- OCR PROCESSING START ---
            ocr_text = ""
            ocr_prob = ""
            try:
                # Crop the detected plate
                plate_crop = frame[ymin:ymax, xmin:xmax]
                
                # Check if crop is valid
                if plate_crop.size > 0:
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (1, 1), 10)
                    structuring_element = np.zeros((40, 40), np.uint8)
                    structuring_element[1:-1, 1:-1] = 1
                    final_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuring_element)
                    
                    ocr_results = reader.readtext(final_img, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    
                    # Iterate through results and check confidence
                    for (bbox, text, prob) in ocr_results:
                        ocr_prob = round(prob, 2)
                        # Only keep text if probability is higher than your threshold
                        if prob >= 0.98:
                            ocr_text += text
                            
                    ocr_text = ocr_text.strip()

                    # Discard prediction if text is too short
                    if len(ocr_text) < 6:
                        ocr_text = ""
                    else:
                        # --- SAVE LOGIC ---
                        # Use regex to keep only alphanumeric for safe filename
                        safe_text = re.sub(r'[^a-zA-Z0-9-]', '', ocr_text)
                        if safe_text:
                            # Add a timestamp to prevent overwriting
                            timestamp = int(time.time())
                            img_name = os.path.join(save_dir, f'{safe_text}_{timestamp}.jpg')
                            cv2.imwrite(img_name, final_img)
                        # --- SAVE LOGIC END ---

            except Exception as e:
                print(f"OCR Error: {e}")
            # --- OCR PROCESSING END ---

            # --- DRAWING LOGIC ---
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            
            # Update label to include OCR text if available
            label = f'{classname}: {int(conf*100)}%'
            if ocr_text:
                label += f' | {ocr_text}'
                label += f' | {ocr_prob}'

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
            
            object_count = object_count + 1
            # --- DRAWING LOGIC END ---

    # Calculate and draw framerate
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results',frame) 
    
    if record: recorder.write(frame)

    # Key press handling
    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): 
        break
    elif key == ord('s') or key == ord('S'): 
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): 
        cv2.imwrite('capture.png', frame)
    
    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
