import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import sys
import os

# ✅ Ensure parent directory is in the import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from tracker import Tracker

# ==========================================
# PATH CONFIGURATION
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8s.pt')
COCO_PATH = os.path.join(BASE_DIR, 'utils', 'coco.names')
VIDEO_PATH = os.path.join(BASE_DIR, 'data', 'angelo.mp4')
OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'output_angelo.avi')

# ==========================================
# MODEL AND SETUP
# ==========================================
model = YOLO(MODEL_PATH)

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture(VIDEO_PATH)

# Get actual video resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer (use actual size)
output = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

# Load class names
with open(COCO_PATH, 'r') as file:
    class_list = file.read().strip().split('\n')

count = 0
tracker = Tracker()

# ==========================================
# RECTANGLE ZONES - FIXED COORDINATES
# ==========================================
# Define zones that make sense for your video
# Format: (x1, y1, x2, y2)
entry_zone = (50, 200, 300, frame_height - 100)   # Left side zone
exit_zone = (frame_width - 300, 200, frame_width - 50, frame_height - 100)  # Right side zone

# ==========================================
# TRACKING AND COUNTING VARIABLES
# ==========================================
tracked_objects = {}  # Stores object state: {id: {'entry_seen': bool, 'exit_seen': bool, 'counted_enter': bool, 'counted_exit': bool}}
enter_counter = set()  # Use sets to avoid duplicates
exit_counter = set()

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    detections = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row[:6])
        if 'person' in class_list[class_id]:
            detections.append([x1, y1, x2, y2])

    bbox_id = tracker.update(detections)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

        # Initialize tracking for new object
        if id not in tracked_objects:
            tracked_objects[id] = {
                'entry_seen': False,
                'exit_seen': False,
                'counted_enter': False,
                'counted_exit': False
            }

        # Check if person is in entry zone (left side)
        if entry_zone[0] < cx < entry_zone[2] and entry_zone[1] < cy < entry_zone[3]:
            tracked_objects[id]['entry_seen'] = True
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Red for entry zone
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)

        # Check if person is in exit zone (right side)
        if exit_zone[0] < cx < exit_zone[2] and exit_zone[1] < cy < exit_zone[3]:
            tracked_objects[id]['exit_seen'] = True
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Green for exit zone
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)

        # COUNTING LOGIC:
        # Person ENTERED: First seen in exit zone, then seen in entry zone
        if (tracked_objects[id]['exit_seen'] and 
            entry_zone[0] < cx < entry_zone[2] and entry_zone[1] < cy < entry_zone[3] and
            not tracked_objects[id]['counted_enter']):
            
            enter_counter.add(id)
            tracked_objects[id]['counted_enter'] = True
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 3)  # Blue for counted enter
            cvzone.putTextRect(frame, f'ENTER {id}', (x3, y3-30), 1, 2)

        # Person EXITED: First seen in entry zone, then seen in exit zone  
        if (tracked_objects[id]['entry_seen'] and 
            exit_zone[0] < cx < exit_zone[2] and exit_zone[1] < cy < exit_zone[3] and
            not tracked_objects[id]['counted_exit']):
            
            exit_counter.add(id)
            tracked_objects[id]['counted_exit'] = True
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 3)  # Yellow for counted exit
            cvzone.putTextRect(frame, f'EXIT {id}', (x3, y3-30), 1, 2)

    # Draw entry and exit rectangles
    cv2.rectangle(frame, (entry_zone[0], entry_zone[1]), (entry_zone[2], entry_zone[3]), (0, 0, 255), 2)
    cv2.rectangle(frame, (exit_zone[0], exit_zone[1]), (exit_zone[2], exit_zone[3]), (0, 255, 0), 2)
    cvzone.putTextRect(frame, 'Entry Zone', (entry_zone[0] + 10, entry_zone[1] - 30), 1, 2)
    cvzone.putTextRect(frame, 'Exit Zone', (exit_zone[0] - 100, exit_zone[1] - 30), 1, 2)

    # Show counts
    downcount, upcount = len(enter_counter), len(exit_counter)
    cvzone.putTextRect(frame, f'Entered: {downcount}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'Exited: {upcount}', (50, 160), 2, 2)

    output.write(frame)
    cv2.imshow('RGB', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
output.release()
cv2.destroyAllWindows()

print(f"Final Count - Entered: {len(enter_counter)}, Exited: {len(exit_counter)}")