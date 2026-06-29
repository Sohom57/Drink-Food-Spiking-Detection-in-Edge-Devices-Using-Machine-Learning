# 👁️‍🗨️ Owner-Aware Personal Space Monitoring for Drink/Food Spiking Detection using Edge-AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0%2B-green.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0%2B-red.svg)](https://opencv.org/)
[![InsightFace](https://img.shields.io/badge/InsightFace-0.7.3-orange.svg)](https://github.com/deepinsight/insightface)

A modular, high-performance computer vision pipeline designed for real-time Digital Image Processing. This system simultaneously tracks hands, authenticates faces, and detects specific objects (like food and utensils) using a hybrid architecture of MediaPipe, InsightFace, and ONNX-optimized YOLO models.

---

## ✨ Interactive Feature Breakdown

*Click on any section below to expand and learn more about the specific modules.*

<details>
<summary><b>✋ Multi-Hand Tracking (MediaPipe)</b></summary>
Tracks up to 6 hands simultaneously in real-time. It maps 21 3D landmarks per hand, drawing customized skeletal connections and highlighting key fingertips and wrists.
</details>

<details>
<summary><b>👤 Face Authentication (InsightFace + MediaPipe)</b></summary>
Uses MediaPipe for lightning-fast face localization, passing cropped regions to an InsightFace `buffalo_l` model on a background thread for seamless facial recognition and authentication against a master `user.jpg` profile.
</details>

<details>
<summary><b>🍽️ Object & Food Detection (YOLOv8 ONNX)</b></summary>
A CPU-optimized YOLOv8 Nano model configured specifically to detect dining-related items (bottles, cups, forks, knives, spoons, bowls) with custom Non-Maximum Suppression (NMS) handling.
</details>

<details>
<summary><b>⚡ Automated Model Management</b></summary>
Automatically fetches required `.task` and `.tflite` models directly from Google Cloud storage on the first run, ensuring a frictionless setup process.
</details>

---

## 📂 Project Folder Structure

To ensure the pipeline runs correctly, your repository and local environment should be structured as follows:

```text
📦 Vision-Tracking-Pipeline
 ┣ 📜 main.py                  # Main entry point and OpenCV rendering loop
 ┣ 📜 module_hands.py          # Hand tracking class (HandTracker)
 ┣ 📜 module_faces.py          # Face detection & auth class (FaceTracker)
 ┣ 📜 module_objects.py        # ONNX YOLOv8 food/object detection (FoodDetector)
 ┣ 📜 module_metrics.py        # Performance and FPS evaluation
 ┣ 📜 utils_download.py        # Auto-downloader for heavy AI model weights
 ┣ 📜 requirements.txt         # Python dependencies
 ┣ 🖼️ user.jpg                 # Master identity image for face authentication
 ┃ 
 ┣ 🗃️ (Auto-Downloaded Models) # These will appear after running main.py:
 ┃ ┣ 📜 hand_landmarker.task   
 ┃ ┣ 📜 face_landmarker.task   
 ┃ ┗ 📜 efficientdet.tflite    
 ┃
 ┗ 🗃️ (Manual Model)
   ┗ 📜 yolov8n.onnx           # Must be placed in root prior to object detection
```

---

## ⚙️ Path Setup & Installation

Follow these steps to configure your environment and run the pipeline.

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment. Install the required libraries via the provided requirements file:

```bash
python -m pip install -r requirements.txt
```

### 2. Prepare the Authentication Target
Place an image of the target user in the root directory and name it **`user.jpg`**. The `FaceTracker` module uses this specific path to generate the baseline embedding for authentication.

### 3. Model Weights Setup
* **ONNX Model:** Ensure the `yolov8n.onnx` file is present in the root directory alongside your scripts.
* **MediaPipe Models:** You do not need to download the `.task` files manually! The `utils_download.py` script is triggered automatically inside `main.py` and will download `hand_landmarker.task`, `face_landmarker.task`, and `efficientdet.tflite` to your root directory upon the first execution.

---

## 🚀 Usage

Simply execute the main script. The system will verify model files, initialize the camera (`cv2.VideoCapture(0)`), and launch the real-time tracking dashboard.

```bash
python main.py
```

**Controls:**
* Press **`q`** inside the video window to safely terminate the pipeline and close background threads.

---

## 🛠️ Performance Notes

This project heavily utilizes `CPUExecutionProvider` for ONNX and InsightFace to maximize compatibility across machines without dedicated GPUs. Heavy tasks, such as generating face embeddings, are isolated on separate Python threads to prevent the main OpenCV rendering loop from dropping frames.
EOF
