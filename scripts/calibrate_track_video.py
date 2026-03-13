import cv2
import json
import numpy as np
import os

# -----------------------------
# Select video
# -----------------------------
video_path = "./videos/video2.mp4"

# automatically extract video name
video_name = os.path.splitext(os.path.basename(video_path))[0]

roi_path = f"./calibrations/{video_name}_roi.json"

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")


# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

if not ret:
    print("❌ Error reading video")
    exit()

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_callback)

print("\n===================================")
print(" Railway Track Calibration Tool")
print("===================================")
print(f"Video: {video_name}")
print("Click points along the railway track.")
print("Press ESC when done.\n")

while True:

    temp = frame.copy()

    for p in points:
        cv2.circle(temp, p, 5, (0,255,0), -1)

    if len(points) > 1:
        pts = np.array(points, np.int32)
        cv2.polylines(temp, [pts], False, (255,0,0), 2)

    cv2.imshow("Calibration", temp)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()

# -----------------------------
# Save ROI
# -----------------------------
with open(roi_path, "w") as f:
    json.dump(points, f)

print(f"\n✅ ROI saved to: {roi_path}\n")

cap.release()