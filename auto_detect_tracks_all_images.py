from ultralytics import YOLO
import cv2
import numpy as np
import os

# -----------------------
# Load YOLO model
# -----------------------
model = YOLO("yolov8n.pt")

image_folder = "images"


# -----------------------
# Detect railway track area automatically
# -----------------------
def detect_track_roi(image):

    height, width = image.shape[:2]

    # Use only bottom part of image
    roi_frame = image[int(height*0.4):height, :]

    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=80,
        minLineLength=150,
        maxLineGap=50
    )

    if lines is None:
        return None

    filtered_lines = []

    for line in lines:

        x1,y1,x2,y2 = line[0]

        angle = np.arctan2((y2-y1),(x2-x1)) * 180 / np.pi

        # remove horizontal / vertical lines
        if abs(angle) < 20:
            continue
        if abs(angle) > 160:
            continue

        filtered_lines.append((x1,y1,x2,y2))

    if len(filtered_lines) < 2:
        return None

    # choose two longest lines
    filtered_lines = sorted(
        filtered_lines,
        key=lambda l: abs(l[1]-l[3]),
        reverse=True
    )

    l1 = filtered_lines[0]
    l2 = filtered_lines[1]

    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    # adjust because we cropped image earlier
    offset = int(height*0.4)

    roi = np.array([
        (x1,y1+offset),
        (x2,y2+offset),
        (x4,y4+offset),
        (x3,y3+offset)
    ], np.int32)

    return roi


# -----------------------
# Check bounding box intersection
# -----------------------
def box_intersects_roi(box, roi):

    x1,y1,x2,y2 = box

    corners = [
        (x1,y1),
        (x2,y1),
        (x1,y2),
        (x2,y2)
    ]

    for corner in corners:
        if cv2.pointPolygonTest(roi, corner, False) >= 0:
            return True

    return False


# -----------------------
# Process all images
# -----------------------
for img_name in os.listdir(image_folder):

    if not img_name.lower().endswith(".jpg"):
        continue

    path = os.path.join(image_folder,img_name)

    image = cv2.imread(path)

    print("\nProcessing:", img_name)

    roi = detect_track_roi(image)

    if roi is None:
        print("Track not detected")
        continue

    results = model(image)

    for box in results[0].boxes:

        cls_id = int(box.cls[0])
        name = model.names[cls_id]

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        intersects = box_intersects_roi((x1,y1,x2,y2),roi)

        if intersects and name != "train":

            print("ALERT:",name,"on track")

            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)

            cv2.putText(
                image,
                f"ALERT:{name}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

        else:

            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

    # draw ROI
    cv2.polylines(image,[roi],True,(255,0,0),2)

    cv2.imshow("Detection",image)

    cv2.waitKey(0)

cv2.destroyAllWindows()