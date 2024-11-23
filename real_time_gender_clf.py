import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("C:\\Users\\ASUS\\PycharmProjects\\face_detection_web\\gender_classification_model.h5")

# Define a function to preprocess the image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64, 64, 1) / 255.0
    return img

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for prediction
    preprocessed = preprocess_image(frame)

    # Make a prediction
    prediction = model.predict(preprocessed)
    gender = "Male" if prediction[0][0] > 0.5 else "Female"

    # Display the result on the frame
    cv2.putText(frame, f'Gender: {gender}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Gender Detection', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
