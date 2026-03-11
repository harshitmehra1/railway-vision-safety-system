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
# Load model
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

roi = np.array(roi_points, np.int32)

# -----------------------------
# Alert settings
# -----------------------------
alert_cooldown = 3
last_alert_time = {}

animal_classes = ["dog","cat","cow","horse","sheep","elephant"]
object_classes = ["backpack","suitcase","handbag","bag"]

# -----------------------------
# Intersection check
# -----------------------------
def box_intersects_roi(box, roi):

    x1,y1,x2,y2 = box

    corners = [
        (x1,y1),
        (x2,y1),
        (x1,y2),
        (x2,y2)
    ]

    for c in corners:
        if cv2.pointPolygonTest(roi, c, False) >= 0:
            return True

    return False


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

    results = model(frame, verbose=False)

    for box in results[0].boxes:

        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        intersects = box_intersects_roi((x1,y1,x2,y2), roi)

        if name == "train":
            continue

        color = (0,255,0)
        label = name.upper()

        if intersects:

            color = (0,0,255)

            # classify groups
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

                print(f"[ ALERT ] {label} detected on railway track ({name}) ")

                last_alert_time[label] = current_time

            cv2.putText(
                frame,
                "⚠ OBSTACLE ON TRACK",
                (40,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # draw track polygon
    cv2.polylines(frame,[roi],True,(255,0,0),2)

    cv2.imshow("Railway Safety System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

print("\nMonitoring stopped.\n")