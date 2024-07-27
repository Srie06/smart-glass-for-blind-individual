import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

padding = 30
imgSize = 300
counter = 0
video_folder = "captures"

def get_gesture_name():
    while True:
        gesture_name = input("Enter the gesture name: ").strip()
        if not gesture_name:
            print("Gesture name cannot be empty.")
            continue
        
        sanitized_name = re.sub(r'[<>:"/\\|?*]', '', gesture_name)
        if not sanitized_name:
            print("Gesture name contains only invalid characters.")
            continue
        
        data_folder = os.path.join(video_folder, sanitized_name)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            return data_folder
        else:
            print(f"Folder '{data_folder}' already exists. Please enter a different name.")

folder = get_gesture_name()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    hands, img = detector.findHands(img)

    if hands:
        for i, hand in enumerate(hands):
            x, y, w, h = hand['bbox']
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x + w + padding, img.shape[1])
            y_end = min(y + h + padding, img.shape[0])

            hand_img = img[y_start:y_end, x_start:x_end]
            hand_h, hand_w, _ = hand_img.shape

            scale = min(imgSize / hand_h, imgSize / hand_w)
            new_w, new_h = int(hand_w * scale), int(hand_h * scale)
            hand_resized = cv2.resize(hand_img, (new_w, new_h))

            canvas = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255
            x_offset = (imgSize - new_w) // 2
            y_offset = (imgSize - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = hand_resized

            cv2.imshow(f"Hand Detection {i+1}", canvas)
    
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    
    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break

    if key == ord("s"):
        cv2.imwrite(f"{folder}/{counter}_hand.jpg", canvas)
        counter += 1
        print(f"Saved {counter} sets of images.")
