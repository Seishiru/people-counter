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
VIDEO_PATH = os.path.join(BASE_DIR, 'data', '3.mp4')
OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'output_final.avi')

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

# Output video writer
output = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'XVID'),
    30,
    (1020, 500)
)

# Load class names
with open(COCO_PATH, 'r') as file:
    class_list = file.read().strip().split('\n')

count = 0
persondown = {}
personup = {}
counter1, counter2 = [], []
tracker = Tracker()

cy1, cy2, offset = 194, 220, 6

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

    frame = cv2.resize(frame, (1020, 500))
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

        # Down direction
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            persondown[id] = (cx, cy)

        if id in persondown:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if id not in counter1:
                    counter1.append(id)

        # Up direction
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            personup[id] = (cx, cy)

        if id in personup:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if id not in counter2:
                    counter2.append(id)

    # Lines
    cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

    downcount, upcount = len(counter1), len(counter2)
    cvzone.putTextRect(frame, f'Down: {downcount}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'Up: {upcount}', (50, 160), 2, 2)

    output.write(frame)
    cv2.imshow('RGB', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
output.release()
cv2.destroyAllWindows()
