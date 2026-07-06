# Real-Time Driver Drowsiness Detection System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HimanshuBairwa/Driver-Drowsiness-Detecttion/blob/main/ML_PROJECT.ipynb)

> **🎥 MUST WATCH:** Please check out my **Demo Video** (attached in the releases/issues below), which I recorded myself, to see the real-time drowsiness detection and audio alarm in action!

A computer vision pipeline that detects driver fatigue in real-time. It uses a **MobileNet** transfer-learning binary classifier (Drowsy / Non-Drowsy) trained on face images, deployed in a live webcam loop that tracks consecutive closed-eye frames and plays an audio alarm when drowsiness is detected.

## Key Features

- **Binary Classification (Drowsy/Non-Drowsy)**: Utilizes a custom MobileNet head built via Transfer Learning to classify eye states with high accuracy.
- **Real-Time Webcam Inference**: Captures live video feed and performs frame-by-frame analysis.
- **Haar-Cascade Region Tracking**: Isolates the driver's face and crops the eye region (`haarcascade_frontalface_default.xml`, `haarcascade_eye_tree_eyeglasses.xml`) before passing it to the neural network.
- **Temporal State Tracking & Audio Alerts**: Tracks consecutive frames where eyes are closed; if the fatigue threshold is exceeded, a "Sleep Alert!!" triggers alongside an audio alarm.

## Quickstart

This project is contained entirely within a single Jupyter/Colab notebook for easy execution.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HimanshuBairwa/Driver-Drowsiness-Detecttion.git
   cd Driver-Drowsiness-Detecttion
   ```

2. **Dataset Setup:**
   The project requires the `Driver Drowsiness Dataset (DDD)` containing `Drowsy` and `Non Drowsy` face images. 
   Upload this dataset to your Google Drive under `/content/drive/MyDrive/Driver Drowsiness Dataset (DDD)/` if you are running the notebook on Google Colab.

3. **Install Dependencies (if running locally):**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Code:**
   Open `ML_PROJECT.ipynb` in Google Colab (recommended for built-in JavaScript webcam API support) or Jupyter Notebook and run the cells sequentially. 
   *(Note: Upload an MP3 file when prompted by the audio alert cell to set your custom alarm sound).*

## Technical Implementation Details

- **Architecture**: `MobileNet` backbone truncated at `layers[-4]` + `Flatten` + `Dense(1, sigmoid)`.
- **Loss & Optimizer**: `binary_crossentropy` and `adam`.
- **Image Preprocessing**: Grayscale to RGB conversion, 224x224 resize to match MobileNet input requirements.
- **Webcam Integration**: Employs injected JavaScript to access the browser's `getUserMedia` API seamlessly inside the Colab environment.
