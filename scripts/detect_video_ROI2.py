from ultralytics import YOLO
import cv2
import json
import numpy as np
import time
import os

# -----------------------------
# Select video
# -----------------------------
video_path = "./videos/video2.mp4"

video_name = os.path.splitext(os.path.basename(video_path))[0]
roi_path = f"./calibrations/{video_name}_roi.json"

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("./models/yolov8s.pt")

# -----------------------------
# Load ROI
# -----------------------------
if not os.path.exists(roi_path):
    print(f"❌ ROI file not found: {roi_path}")
    print("Run calibration first.")
    exit()

with open(roi_path) as f:
    roi_points = json.load(f)

roi_original = np.array(roi_points, np.int32)

# -----------------------------
# Expand ROI by 20%
# -----------------------------
def expand_roi(roi, scale=1.2):

    roi = roi.astype(np.float32)
    center = np.mean(roi, axis=0)

    expanded_roi = []

    for point in roi:
        vector = point - center
        new_point = center + vector * scale
        expanded_roi.append(new_point)

    return np.array(expanded_roi, dtype=np.int32)

roi = expand_roi(roi_original, scale=1.2)

# -----------------------------
# Alert settings
# -----------------------------
alert_cooldown = 3
last_alert_time = {}

animal_classes = ["dog","cat","cow","horse","sheep","elephant"]
object_classes = ["backpack","suitcase","handbag","bag"]

# -----------------------------
# Detection memory
# -----------------------------
recent_detections = []

# -----------------------------
# ROI check
# -----------------------------
def box_intersects_roi(box, roi):

    x1,y1,x2,y2 = box

    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)

    return cv2.pointPolygonTest(roi, (cx,cy), False) >= 0

# -----------------------------
# Motion detection setup
# -----------------------------
previous_frame = None

# -----------------------------
# Frame counter
# -----------------------------
frame_count = 0

# -----------------------------
# Start video
# -----------------------------
cap = cv2.VideoCapture(video_path)

print("\n===================================")
print(" Railway Vision Safety System")
print("===================================")
print("Monitoring video stream...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    object_detected = False

    # -----------------------------
    # YOLO detection (every 3 frames)
    # -----------------------------
    if frame_count % 3 == 0:

        results = model(frame, verbose=False)

        for box in results[0].boxes:

            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])

            if conf < 0.45:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            if name == "train":
                continue

            if not box_intersects_roi((x1,y1,x2,y2), roi):
                continue

            object_detected = True

            label = name.upper()

            if name in animal_classes:
                label = "ANIMAL"

            elif name in object_classes:
                label = "OBJECT"

            elif name == "person":
                label = "PERSON"

            current_time = time.time()

            if label not in last_alert_time:
                last_alert_time[label] = 0

            if current_time - last_alert_time[label] > alert_cooldown:

                print(f"[ ALERT ] {label} detected on railway track ({name})")

                last_alert_time[label] = current_time

            recent_detections = [(x1,y1,x2,y2,label)]

    # -----------------------------
    # Draw detections
    # -----------------------------
    for det in recent_detections:

        x1,y1,x2,y2,label = det

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.putText(
            frame,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

        cv2.putText(
            frame,
            "⚠ OBSTACLE ON TRACK",
            (40,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

    # -----------------------------
    # Motion Detection
    # -----------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9,9), 0)

    if previous_frame is None:
        previous_frame = gray.copy()
        continue

    frame_diff = cv2.absdiff(previous_frame, gray)

    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        if cv2.contourArea(contour) < 500:
            continue

        x,y,w,h = cv2.boundingRect(contour)

        if box_intersects_roi((x,y,x+w,y+h), roi) and not object_detected:

            if time.time() - last_alert_time.get("motion",0) > alert_cooldown:

                print("[ MOTION ] Movement detected near railway track")

                last_alert_time["motion"] = time.time()

    previous_frame = gray.copy()

    # -----------------------------
    # Draw ONLY the actual track
    # -----------------------------
    cv2.polylines(frame,[roi_original],True,(255,0,0),2)

    cv2.imshow("Railway Safety System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

print("\nMonitoring stopped.\n")