import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Constants and Configuration ---
DATASET_PATH = "path/to/your/face_dataset"
MODEL_PATH = "path/to/your/tensorflow_model.h5"
LABEL_DICT = {0: "Alice", 1: "Bob", 2: "Cara"}  
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
IMG_SIZE = (224, 224) 
CONFIDENCE_THRESHOLD = 0.7  # Set a minimum confidence threshold for predictions
DETECTION_SCALE_FACTOR = 1.1
DETECTION_MIN_NEIGHBORS = 5

# --- Functions ---
def load_dataset(path):
    """Loads face images and labels from a directory."""
    images, labels = [], []
    for label_dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, label_dir)):
            for img_file in os.listdir(os.path.join(path, label_dir)):
                img_path = os.path.join(path, label_dir, img_file)
                if os.path.isfile(img_path) and img_path.endswith(('.jpg', '.png')):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(int(label_dir))
    return np.array(images), np.array(labels)

def preprocess_image(img):
    """Preprocesses an image for TensorFlow model prediction."""
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img_to_array(img), axis=0)

# --- Main Program ---
if __name__ == "__main__":
    images, labels = load_dataset(DATASET_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    model = load_model(MODEL_PATH)  

    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=DETECTION_SCALE_FACTOR, minNeighbors=DETECTION_MIN_NEIGHBORS)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            label, confidence = recognizer.predict(face_roi)
            if confidence < CONFIDENCE_THRESHOLD:
                opencv_name = "Unknown"
            else:
                opencv_name = LABEL_DICT.get(label, "Unknown")

            face_img = preprocess_image(face_roi)
            predictions = model.predict(face_img)
            deep_label = np.argmax(predictions[0])
            deep_name = LABEL_DICT.get(deep_label, "Unknown")

            text = f"{opencv_name}/{deep_name} Confidence: {confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
