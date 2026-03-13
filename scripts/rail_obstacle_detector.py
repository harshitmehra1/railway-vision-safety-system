from ultralytics import YOLO
import cv2

# -----------------------------
# Load models
# -----------------------------
rail_model = YOLO("./models/best.pt")        # your trained rail detector
object_model = YOLO("./models/yolov8n.pt")   # pretrained object detector


# -----------------------------
# Load image
# -----------------------------
image = cv2.imread("images/a3.jpg")
h, w, _ = image.shape


# -----------------------------
# Detect railway tracks
# -----------------------------
rail_results = rail_model(image)

rails = []

for box in rail_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Ignore detections too high in the image (reduces platform false positives)
    if y2 < h * 0.4:
        continue

    rails.append((x1, y1, x2, y2))


# -----------------------------
# Detect objects
# -----------------------------
obj_results = object_model(image)

objects = []

for box in obj_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    conf = float(box.conf[0])

    name = object_model.names[cls]

    objects.append((name, conf, x1, y1, x2, y2))


# -----------------------------
# Check if object is on track
# -----------------------------
alert_objects = []

for name, conf, x1, y1, x2, y2 in objects:

    # Ignore trains (not an obstacle)
    if name == "train":
        continue

    for rx1, ry1, rx2, ry2 in rails:

        # Check bounding box intersection
        if not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2):

            alert_objects.append((name, conf, x1, y1, x2, y2))
            break


# -----------------------------
# Draw rail boxes
# -----------------------------
for rx1, ry1, rx2, ry2 in rails:
    cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
    cv2.putText(image, "Track", (rx1, ry1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


# -----------------------------
# Draw object boxes
# -----------------------------
for name, conf, x1, y1, x2, y2 in objects:

    color = (0, 255, 0)

    for alert in alert_objects:
        if (name, conf, x1, y1, x2, y2) == alert:
            color = (0, 0, 255)

    label = f"{name} {conf:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# -----------------------------
# Print alerts
# -----------------------------
for name, conf, x1, y1, x2, y2 in alert_objects:
    print(f"ALERT: {name} detected on railway track")


# -----------------------------
# Show result
# -----------------------------
cv2.imshow("Railway Safety System", image)
cv2.waitKey(0)
cv2.destroyAllWindows()