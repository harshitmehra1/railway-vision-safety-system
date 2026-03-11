import cv2
import json
import numpy as np

points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print("Point added:", (x, y))

image = cv2.imread("images/c3.jpg")

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_callback)

while True:
    temp = image.copy()

    # draw points
    for p in points:
        cv2.circle(temp, p, 5, (0,255,0), -1)

    # draw polygon lines
    if len(points) > 1:
        pts = np.array(points, np.int32)
        cv2.polylines(temp, [pts], False, (255,0,0), 2)

    cv2.imshow("Calibration", temp)

    key = cv2.waitKey(1)

    # press ESC when done
    if key == 27:
        break

cv2.destroyAllWindows()

# save ROI
with open("track_roi.json", "w") as f:
    json.dump(points, f)

print("ROI saved to track_roi.json")