from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Ensure the 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the trained model
model = load_model("C:\\Users\\ASUS\\PycharmProjects\\face_detection_web\\gender_classification_model.h5")

# Preprocess the uploaded image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64, 64, 1) / 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    # Save and preprocess the file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    preprocessed = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(preprocessed)
    gender = "Male" if prediction[0][0] > 0.5 else "Female"

    # Remove the uploaded file after prediction
    os.remove(file_path)

    return render_template('result.html', gender=gender)

if __name__ == '__main__':
    app.run(debug=True)
