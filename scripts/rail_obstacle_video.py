from ultralytics import YOLO
import cv2
import time

# ---------------------------
# Load models
# ---------------------------
rail_model = YOLO("best.pt")
object_model = YOLO("yolov8s.pt")

# ---------------------------
# Open video
# ---------------------------
cap = cv2.VideoCapture("video4.mp4")

frame_skip = 2
frame_count = 0

# rail detection every N frames
rail_refresh_rate = 10
rails = []

last_alert_time = 0
alert_cooldown = 2

# how much of frame to consider as track zone
track_region = 0.8

# detection threshold
confidence_threshold = 0.35

# object grouping
animal_classes = ["dog","cat","cow","horse","sheep","elephant","bear"]
vehicle_classes = ["car","truck","bus","motorcycle","bicycle"]

print("\nRailway Safety System Started\n")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    h, w, _ = frame.shape

    # ---------------------------
    # Detect rails occasionally
    # ---------------------------
    if frame_count % rail_refresh_rate == 0:

        rail_results = rail_model(frame, verbose=False)
        rails = []

        for box in rail_results[0].boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # ignore extremely wide detections (platform)
            if (x2 - x1) > w * 0.9:
                continue

            # ignore rails too high in frame
            if y2 < h * (1 - track_region):
                continue

            rails.append((x1, y1, x2, y2))

    # ---------------------------
    # Detect objects
    # ---------------------------
    obj_results = object_model(frame, verbose=False)

    for box in obj_results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < confidence_threshold:
            continue

        name = object_model.names[cls]

        if name == "train":
            continue

        # ignore tiny detections
        if (x2-x1)*(y2-y1) < 600:
            continue

        # classify object group
        label_name = name.upper()

        if name in animal_classes:
            label_name = "ANIMAL"

        if name in vehicle_classes:
            label_name = "VEHICLE"

        color = (0,255,0)

        for rx1, ry1, rx2, ry2 in rails:

            if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):

                color = (0,0,255)

                if time.time() - last_alert_time > alert_cooldown:

                    print(f"[ALERT] {label_name} detected on railway track")

                    last_alert_time = time.time()

                cv2.putText(
                    frame,
                    "⚠ OBSTACLE ON TRACK",
                    (40,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3
                )

                break

        label = f"{name} {conf:.2f}"

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            frame,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ---------------------------
    # Draw rails
    # ---------------------------
    for rx1, ry1, rx2, ry2 in rails:

        cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),(255,0,0),2)

        cv2.putText(
            frame,
            "TRACK",
            (rx1,ry1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,0,0),
            2
        )

    # system label
    cv2.putText(
        frame,
        "Railway Vision Safety System",
        (20, h-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Railway Safety System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nSystem stopped\n")