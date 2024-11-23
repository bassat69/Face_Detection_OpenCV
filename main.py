import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path to the UTKFace dataset
data_dir = "C:\\Users\\ASUS\\Downloads\\archive (16)\\crop_part1"

# Lists to store images and labels
images = []
labels = []

# Loop through all images in the dataset directory
for filename in os.listdir(data_dir):
    if filename.endswith(".jpg"):
        parts = filename.split("_")

        # Ensure there are at least three parts (age, gender, ethnicity)
        if len(parts) >= 3:
            age = parts[0]
            gender = parts[1]
            ethnicity = parts[2]

            # Only proceed if the gender is valid (0 or 1)
            if gender in ['0', '1']:
                image_path = os.path.join(data_dir, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (64, 64))
                images.append(image)
                labels.append(int(gender))  # Gender label: 0 for male, 1 for female

# Convert lists to numpy arrays and normalize pixel values
X = np.array(images).reshape(-1, 64, 64, 1) / 255.0
y = to_categorical(np.array(labels), num_classes=2)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 output neurons for 'male' and 'female'
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model
model.save('gender_classification_model.h5')
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


