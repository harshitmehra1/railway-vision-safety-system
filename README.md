#  🚆 Railway Vision Safety System
AI-based obstacle detection for railway tracks using YOLOv8 and computer vision

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-red)

## Project Overview

The **Railway Vision Safety System** is a computer vision and AI-based project designed to detect obstacles on railway tracks. Utilizing state-of-the-art object detection models, the system monitors video feeds to identify objects that fall within a predefined track area, triggering alerts to help prevent collisions and ensure the safety of railway operations.

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Computer Vision

## Motivation

Railway accidents involving humans, animals, and static objects can lead to severe consequences, including loss of life, injury, and significant damage to railway infrastructure. This project aims to enhance railway safety by providing an early warning system that protects people, animals, and the physical railway networks. 

## How the System Works

The system operates through a straightforward yet effective process:
1. **Detection**: It continuously analyzes video frames to detect various objects using YOLOv8.
2. **Calibration**: The railway track area (Region of Interest - ROI) is defined manually using a dedicated calibration tool.
3. **Intersection Checking**: The system checks if the bounding boxes of detected objects intersect with the calibrated track area.
4. **Alerting**: If an object is detected within the track area, the system triggers alerts for potential obstacles.

### What is Detected?
The system is trained to detect potential hazards, including:
- **People**: Pedestrians or workers on or near the tracks.
- **Animals**: Common animals that may wander onto tracks, such as dogs, cows, horses, elephants, etc.
- **Objects**: Static objects that could derail or damage a train, such as bags, suitcases, and backpacks.

*Note: The detection system is specifically configured to ignore trains themselves, focusing only on anomalous obstacles.*

## System Pipeline

1. **Input Video** ➔ 2. **YOLOv8 Object Detection** ➔ 3. **Track ROI Filtering (Calibration)** ➔ 4. **Obstacle Alert Generation**


## Folder Structure

```text
railway-vision-safety-system/
│
├── calibrations/          # Stores JSON calibration files for defined track areas
├── dataset_tools/         # Scripts for preparing or manipulating datasets
├── datasets/              # Data used for training or testing
├── images/                # Sample images and snapshots
├── models/                # YOLOv8 weights and model files
├── scripts/               # Core Python scripts (detection, calibration, etc.)
├── videos/                # Input video files for testing
└── requirements.txt       # Python dependencies
```

## Installation

### Requirements

This project relies on the following major dependencies:
- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV (`opencv-python`)
- NumPy

### Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone https://github.com/harshitmehra1/railway-vision-safety-system.git
   cd railway-vision-safety-system
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Calibration Instructions

Before running the detection on a new video, you must define the railway track area. Each video has its own saved calibration file stored in the `calibrations` folder.

1. **Run the calibration script** on your target video.
2. **Draw the Polygon**: The script will pause the video, allowing you to click points to draw the track region.
3. **Point Order**: You can click as many points as needed. To define the track area accurately, it is recommended to click in the following order:
   - Start from the **left-bottom** of the track.
   - Move to the **left-top**.
   - Move toward the **right side** of the track.
4. **Curves**: Multiple clicks are encouraged so curves in the railway track can be captured properly.
5. **Finish**: Press **ESC** when calibration is finished.
6. The exact polygon coordinates will be saved as a JSON file in the `calibrations/` directory, specifically named for that video.

## How to Run the Project

### 1. Run Calibration
First, calibrate the track area for your video:
```bash
python scripts/calibrate_track_video.py
```

### 2. Running Detection
Once calibrated, run the main detection script:
```bash
python scripts/detect_video_ROI.py 
# or
python scripts/detect_video_ROI2.py
```
The system will load the corresponding JSON calibration file and begin monitoring the video feed for obstacles intersecting the defined track zone.

## Example Output

When running the detection script, the system will display the video feed with the following visual indicators:
- The **Track ROI** drawn as a colored polygon.
- **Bounding Boxes** around detected people, animals, and objects.
- **Alert Visuals** (e.g., red bounding boxes or warning text) when an object enters the track area.

## Demo

![Demo](demo.gif)

Example detections from the system:

- Animal detected on railway track
- Person detected inside track region
- Object detected blocking track

The system highlights obstacles entering the calibrated track region and triggers alerts in real time.



## Possible Improvements

- Real-time notification integration (e.g., SMS, email, or a web dashboard).
- Distance estimation to calculate the exact distance between the train and the obstacle.
- Night vision and thermal camera support for 24/7 monitoring.
- Integration with PTZ (Pan-Tilt-Zoom) cameras for dynamic tracking.

## Conclusion

The Railway Vision Safety System provides an automated, intelligent layer of security to railway operations. By combining robust AI object detection with customizable, precise track calibration, the system aims to drastically reduce the risk of accidents, protecting lives and infrastructure alike.
