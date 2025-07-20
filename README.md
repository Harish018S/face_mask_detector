## 🧠 Face Mask Detection System

This is a real-time Face Mask Detection system built using **Python**, **OpenCV**, and **TensorFlow/Keras**. It uses a pre-trained CNN model to detect whether a person is wearing a face mask or not using webcam input.

---

### 📁 Project Structure
```
face_mask_detector/
├── detect_mask.py # Main script to run mask detection
├── mask_detector.model # Pre-trained CNN model
├── haarcascade_frontalface_default.xml # Haar Cascade for face detection
├── deploy.prototxt # Face detection model config (OpenCV SSD)
├── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained face detector
├── requirements.txt # Python dependencies


---

### ⚙️ Requirements

bash
pip install -r requirements.txt

---

### 🚀 How to Run

bash
python detect_mask.py

---

### 📌 Features

• Real-time webcam detection
• Detects with-mask and without-mask faces
• Uses Haar Cascade and ResNet SSD for face detection
• Built using Python, OpenCV, and TensorFlow

🙏 Acknowledgements
The file mask_detector.model was sourced from chandrikadeb7/Face-Mask-Detection and is licensed under the MIT License.
