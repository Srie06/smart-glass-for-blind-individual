# test_model.py

import os
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import json

# Load the trained model and label map
model = tf.keras.models.load_model('hand_gesture_model.h5')

with open('label_map.json', 'r') as f:
    label_map = json.load(f)

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

imgSize = 300

def preprocess_frame(frame, img_size):
    """Resize and normalize the frame."""
    hand_img = cv2.resize(frame, (img_size, img_size))
    hand_img = hand_img / 255.0
    return np.expand_dims(hand_img, axis=0)

print("Press 'q' to quit the video stream")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    hands, img = detector.findHands(img)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']

            # Define the bounding box with padding
            x_start = max(x - 30, 0)
            y_start = max(y - 30, 0)
            x_end = min(x + w + 30, img.shape[1])
            y_end = min(y + h + 30, img.shape[0])

            # Crop the hand region
            hand_img = img[y_start:y_end, x_start:x_end]

            # Preprocess the image
            preprocessed_img = preprocess_frame(hand_img, imgSize)

            # Predict gesture
            predictions = model.predict(preprocessed_img)
            predicted_class = np.argmax(predictions[0])
            gesture_name = label_map[str(predicted_class)]

            # Display the predicted gesture
            cv2.putText(img, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the original image with predictions
    cv2.imshow("Hand Gesture Recognition", img)

    key = cv2.waitKey(1)
    
    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break
