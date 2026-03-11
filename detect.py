from ultralytics import YOLO
import cv2
import json
import numpy as np

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Load ROI polygon
# -----------------------------
with open("track_roi.json") as f:
    roi_points = json.load(f)

roi = np.array(roi_points, np.int32)

# -----------------------------
# Function to check if bounding box intersects track ROI
# -----------------------------
def box_intersects_roi(box, roi):

    x1, y1, x2, y2 = box

    # corners of bounding box
    corners = [
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2)
    ]

    # check if any corner lies inside ROI
    for corner in corners:
        if cv2.pointPolygonTest(roi, corner, False) >= 0:
            return True

    return False


# -----------------------------
# Load image
# -----------------------------
image = cv2.imread("images/c3.jpg")

# -----------------------------
# Run YOLO detection
# -----------------------------
results = model(image)

# -----------------------------
# Process detections
# -----------------------------
for box in results[0].boxes:

    cls_id = int(box.cls[0])
    name = model.names[cls_id]

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # draw center for debugging
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
    cv2.circle(image, (cx,cy), 5, (255,0,255), -1)

    intersects = box_intersects_roi((x1,y1,x2,y2), roi)

    print("Object:", name, "Box:", (x1,y1,x2,y2), "Intersects ROI:", intersects)

    # ignore trains
    if intersects and name != "train":

        print("ALERT:", name, "on track")

        cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 2)

        cv2.putText(
            image,
            f"ALERT:{name}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

    else:

        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)


# -----------------------------
# Draw track ROI
# -----------------------------
cv2.polylines(image, [roi], True, (255,0,0), 2)

# -----------------------------
# Show result
# -----------------------------
cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()