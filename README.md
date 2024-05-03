# Jetson Nano Face Recognition

This repository contains a Python script for face recognition using OpenCV and TensorFlow on the NVIDIA Jetson Nano. The script utilizes both a traditional machine learning model (LBPHFaceRecognizer) from OpenCV and a deep learning model from TensorFlow to enhance accuracy and robustness.

## Description

The script performs real-time face recognition using a webcam. It employs a pre-trained Haar Cascade model for detecting faces in video frames and then recognizes identities using both an OpenCV LBPH recognizer and a TensorFlow neural network model.

## Features

- **Dual Recognition System:** Uses both OpenCV's LBPH algorithm and TensorFlow's deep learning models for improved accuracy.
- **Real-Time Detection:** Performs face detection and recognition in real-time using a webcam.
- **Easy Customization:** Easy to add or modify pre-trained models and identities.

## Prerequisites

Before you can run this script, make sure you have the following installed:
- Python 3.x
- OpenCV-Python
- TensorFlow 2.x

These can be installed via pip with the following commands:

```bash
pip install numpy opencv-python tensorflow
