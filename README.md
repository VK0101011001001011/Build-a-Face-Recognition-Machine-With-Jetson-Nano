# Jetson Nano Face Recognition 🚀

This repository contains a Python script for face recognition using OpenCV and TensorFlow on the NVIDIA Jetson Nano. The script utilizes both a traditional machine learning model (LBPHFaceRecognizer) from OpenCV and a deep learning model from TensorFlow to enhance accuracy and robustness.

## 📖 Description

The script performs real-time face recognition using a webcam. It employs a pre-trained Haar Cascade model for detecting faces in video frames and then recognizes identities using both an OpenCV LBPH recognizer and a TensorFlow neural network model.

## ✨ Features

- **Dual Recognition System:** Uses both OpenCV's LBPH algorithm and TensorFlow's deep learning models for improved accuracy.
- **Real-Time Detection:** Performs face detection and recognition in real-time using a webcam.
- **Easy Customization:** Easy to add or modify pre-trained models and identities.

## 📋 Prerequisites

Before you can run this script, ensure the following are installed:
- Python 3.x
- OpenCV-Python
- TensorFlow 2.x

These can be installed via pip with the following commands:

```bash
pip install numpy opencv-python tensorflow

```
## Installation

1. **Clone the Repository:**


2. **Setup Your Dataset:**
- Organize your face images in folders named by label (e.g., `0`, `1`, `2`, etc.) inside a directory.
- Each folder should contain images of one person.

3. **Configure the Script:**
- Open the `jetson_nano_face_recognition.py` file.
- Modify the `dataset_path` and `model_path` variables to point to your dataset and TensorFlow model respectively.

## Usage

To run the face recognition script, navigate to the repository folder in your terminal and execute: 
```bash
python jetson_nano_face_recognition.py
```


## How It Works

- **Face Detection:** The script first detects faces in each frame of the webcam feed using a Haar Cascade Classifier.
- **Face Recognition:** Extracted faces are then recognized using both LBPH (Local Binary Patterns Histograms) and a pre-trained TensorFlow model.
- **Display Results:** The script displays the video feed with detected faces highlighted and labeled with the identified names and confidence scores.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the OpenCV and TensorFlow communities for providing excellent libraries and documentation.

Enjoy enhancing your Jetson Nano projects with powerful face recognition capabilities!



