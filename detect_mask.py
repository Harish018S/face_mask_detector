import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("mask_detector.model")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)

labels_dict = {0: 'No Mask', 1: 'Mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (224, 224))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        
        result = model.predict(reshaped)
        print("Prediction score:", result[0][0])  

        
        label = 0 if result[0][0] < 0.5 else 1


        cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.putText(frame, f"{labels_dict[label]} ({result[0][0]:.2f})", (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_dict[label], 2)


    cv2.imshow("Face Mask Detector", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
