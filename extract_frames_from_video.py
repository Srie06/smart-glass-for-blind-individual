# extract_frames_from_video.py

import os
import cv2
import numpy as np

def extract_frames_from_video(video_path, img_size, output_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Resize and normalize the frame
        frame_resized = cv2.resize(frame, (img_size, img_size))
        frame_normalized = frame_resized / 255.0
        
        # Save the frame as an image
        cv2.imwrite(os.path.join(output_folder, f"{count}.jpg"), (frame_normalized * 255).astype(np.uint8))
        count += 1

    cap.release()

# Set parameters
video_folder = "captures"
gesture_name = input("Enter the gesture name to process: ")
video_path = f"{video_folder}/{gesture_name}/output.avi"
output_folder = f"{video_folder}/{gesture_name}/frames"
os.makedirs(output_folder, exist_ok=True)

# Extract frames
extract_frames_from_video(video_path, 300, output_folder)
print(f"Frames extracted and saved in '{output_folder}'")
