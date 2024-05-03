# Jetson Nano Face Recognition 🚀

Welcome to the comprehensive repository designed to unleash the power of face recognition using OpenCV and TensorFlow on the NVIDIA Jetson Nano. This resourceful script capitalizes on advanced machine learning techniques, combining the robustness of OpenCV’s LBPHFaceRecognizer and the cutting-edge capabilities of TensorFlow models.

## 📖 Description

This script facilitates real-time face recognition through a webcam, making it a perfect tool for various applications from security systems to user authentication. Leveraging a pre-trained Haar Cascade model, the script adeptly detects faces in video streams. Post-detection, it employs dual recognition systems—OpenCV’s LBPH algorithm and TensorFlow’s neural network models—to accurately identify individuals.

## ✨ Features

- **Dual Recognition System:** Integrates the reliability of OpenCV’s LBPHFaceRecognizer and the advanced learning capabilities of TensorFlow's models.
- **Real-Time Detection and Recognition:** Seamlessly detects and identifies faces in real-time.
- **Modularity:** Easily adaptable for different use cases and datasets.

## 📋 Prerequisites

To run this script, ensure your environment is set up with the following:
- Python 3.6 or higher
- OpenCV-Python
- TensorFlow 2.x

You can install these dependencies via pip:

    pip install numpy opencv-python tensorflow

## 🛠 Installation

1. **Clone the Repository:**
   Clone this repository to your local machine to get started.
   
    git clone https://github.com/VellVoid/Build-a-Face-Recognition-Machine-With-Jetson-Nano.git
    cd VellVoid/Build-a-Face-Recognition-Machine-With-Jetson-Nano

2. **Prepare the Dataset:**
   Organize your dataset with folders labeled numerically for each individual.

3. **Configure Your Environment:**
   Set the environment variables to point to your dataset and model directories.
   
    export DATASET_PATH=/path/to/your/dataset
    export MODEL_PATH=/path/to/your/model

## 🚀 Usage

To run the face recognition script, navigate to the repository folder and execute:

    python jetson_nano_face_recognition.py

## 🧠 How It Works

- **Face Detection:** Uses a Haar Cascade Classifier to detect faces within the video frames.
- **Recognition Process:** Faces are analyzed using both LBPH and TensorFlow models to match against known identities.
- **Result Display:** Outputs the video with bounding boxes and labels indicating identified persons and their confidence levels.

## 📜 License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 📚 Additional Resources

- [OpenCV Official Tutorials](https://opencv.org/docs/)
- [TensorFlow Full Course](https://www.tensorflow.org/tutorials)
- [NVIDIA Developer Resources for Jetson Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
- [DIY Project on Instructables](https://www.instructables.com/Build-a-Face-Recognition-Machine-With-Jetson-Nano/): Follow this detailed guide to build your own face recognition machine with Jetson Nano.

## 🎉 Acknowledgments

Special thanks to:
- The OpenCV and TensorFlow teams for their exceptional support and resources.
- NVIDIA for making high-performance computing accessible for developers and researchers.
- The creator of the Instructables DIY project for providing a practical and informative guide.

**Explore the possibilities of enhancing your projects with state-of-the-art face recognition technology using the Jetson Nano!**
