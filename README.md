🧠 Face Mask Detection System
This is a real-time Face Mask Detection system built using Python, OpenCV, and TensorFlow/Keras. It uses a pre-trained CNN model to detect whether a person is wearing a face mask or not using webcam input.

📁 Project Structure
face_mask_detector/
├── detect_mask.py                  # Main script to run mask detection
├── mask_detector.model             # Pre-trained CNN model
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── deploy.prototxt                 # Face detection model config (OpenCV SSD)
├── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained face detector
├── requirements.txt                # Python dependencies
⚙️ Requirements


Install the dependencies:
pip install -r requirements.txt


🚀 How to Run
Run the detection system:
python detect_mask.py


It will start your webcam and begin detecting faces with or without masks in real-time.

📌 Features:

Real-time detection using webcam
Detects with-mask and without-mask faces
Uses a custom-trained CNN model
Combines Haar Cascade and ResNet SSD for face detection

