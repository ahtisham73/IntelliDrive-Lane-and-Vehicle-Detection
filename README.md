# YOLOv8 Lane and Vehicle Detection

![YOLOv8](https://img.shields.io/badge/YOLOv8-ObjectDetection-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Introduction
This project implements lane and vehicle detection using the state-of-the-art YOLOv8 (You Only Look Once, version 8) model. The primary goal is to identify vehicles and lane boundaries in video data, which is essential for autonomous driving applications. The YOLOv8 model, with its pretrained weights, enables high accuracy and computational efficiency, making it suitable for real-time perception tasks.

---

## Highlights

- **YOLOv8**: Utilized for vehicle detection with pretrained weights.
- **Lane Detection**: Pipeline to identify and visualize lane boundaries using edge detection and line fitting.
- **Distance Estimation**: Approximate distances of vehicles from the camera using bounding box dimensions.
- **Real-Time Processing**: Handles video data at a target frame rate of 30 FPS.
- **Autonomous Driving Applications**: Supports perception systems for lane-keeping and collision avoidance.

---

## Methodology

### 1. **Lane Detection**
- **Edge Detection**: Applied Canny edge detection to identify edges in the video frame.
- **Region of Interest (ROI)**: Masked the area of interest to focus on lanes.
- **Line Fitting**: Used Hough Line Transformation and polynomial fitting to detect left and right lane boundaries.
- **Visualization**: Created a filled polygon between lane lines for better visualization.

### 2. **Vehicle Detection**
- **YOLOv8 Model**: Leveraged pretrained YOLOv8 weights to detect vehicles.
- **Bounding Boxes**: Drew bounding boxes around detected vehicles.
- **Classification**: Focused on detecting cars with confidence scores ≥ 0.5.

### 3. **Distance Estimation**
- Used bounding box dimensions to estimate the approximate distance of vehicles.
- Assumed a focal length and known vehicle dimensions for basic distance calculation.

### 4. **Integration**
- Combined lane detection and vehicle detection outputs for simultaneous visualization in real time.

---

## Results

The YOLOv8 model successfully detected vehicles and lane boundaries in real-world video data. The system maintained high accuracy and processing speed, even in dynamic environments. Below are key results:

- **Vehicle Detection Accuracy**: High precision for vehicle classification.
- **Lane Detection**: Accurate identification of lane boundaries.
- **Real-Time Processing**: Achieved target frame rate of 30 FPS.

### Screenshots and Demo Video

![Lane Detection Example](placeholder-for-image-1)
![Vehicle Detection Example](placeholder-for-image-2)

[Watch Demo Video](placeholder-for-video-link)

---

## Files and Directory Structure

```
YOLOv8-Lane-Vehicle-Detection/
|-- weights/
|   |-- yolov8n.pt               # Pretrained YOLOv8 model weights
|-- video/
|   |-- video.mp4                # Input video file for detection
|-- lane_vehicle_detection.py    # Main script for detection
|-- README.md                    # Project documentation
```

---

## How to Use

### 1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/YOLOv8-Lane-Vehicle-Detection.git
cd YOLOv8-Lane-Vehicle-Detection
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Download Pretrained Weights**
Place the `yolov8n.pt` weights file in the `weights/` directory. You can download the weights from the [Ultralytics YOLOv8 Repository](https://github.com/ultralytics/ultralytics).

### 4. **Run the Detection**
```bash
python lane_vehicle_detection.py
```

### 5. **Output**
- Detected lane and vehicle visualizations will be displayed in a window.
- Press `q` to exit the visualization.

---

## Conclusion
The pretrained YOLOv8 model effectively detected vehicles and lanes, demonstrating its reliability for real-time autonomous driving perception systems. The pipeline is efficient and adaptable, serving as a foundation for more advanced autonomous vehicle applications. Future improvements could include:

- Integration with planning and control algorithms.
- Camera calibration for improved distance estimation.
- Enhanced robustness for challenging weather conditions.

---

## Future Work
- Support for additional object categories.
- Multi-camera support for improved perception.
- Optimized pipeline for edge devices like Raspberry Pi or NVIDIA Jetson.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)

---
