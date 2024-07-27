# train_model.py

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json
import pickle

# Load and preprocess images for training
def load_data(data_dir, img_size):
    images = []
    labels = []
    label_map = {}
    for idx, folder in enumerate(os.listdir(data_dir)):
        label_map[idx] = folder
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(idx)
            else:
                print(f"Unable to read image {img_path}")
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, label_map

# Set parameters
imgSize = 300
data_dir = 'captures'

images, labels, label_map = load_data(data_dir, imgSize)
images = images / 255.0 

x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define and compile the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(imgSize, imgSize, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_val, y_val))

# Save model and metadata
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

val_loss, val_acc = model.evaluate(x_val, y_val)
print(f'Validation accuracy: {val_acc * 100:.2f}%')

model.save('hand_gesture_model.h5')
print('Model saved as hand_gesture_model.h5')

with open('label_map.json', 'w') as f:
    json.dump(label_map, f)
print('Label map saved as label_map.json')
