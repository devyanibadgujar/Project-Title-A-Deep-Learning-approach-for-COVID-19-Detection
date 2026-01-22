##NEW CODE :: 

from __future__ import division, print_function
import sys
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Define a Flask app
app = Flask(__name__)

# Path to your model
MODEL_PATH = 'E:/COVID-19 DETECTION/Covid19-Detection-Seqential model/Final Working Project/New_Sequential_3.keras'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()  # Necessary for Keras

print('Model loaded. Start serving...')

def model_predict(img_path, model):
    try:
        # Load the image and resize to model's expected input size
        # img = image.load_img(img_path, target_size=(150, 150))  # Adjust size if needed
        # img_array = image.img_to_array(img)
        # img_array = img_array / 255.0  # Normalize the image
        # img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        img = cv2.imread(img_path)
        img = cv2.resize(img,(150,150))
        img_array = np.array(img)
        img_array.shape

        img_array = img_array.reshape(1,150,150,3)
        img_array.shape
        # Make predictions
        a=model.predict(img_array)
        indices = a.argmax()
        indices
        if indices==0:
            return "COVID"
        else:
            return "Normal"
    except Exception as e:
        print(f"Error in model prediction: {e}")
        raise

# Route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Additional routes for static pages
@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/doctor', methods=['GET'])
def doctor():
    return render_template('doctor.html')

@app.route('/testimonial', methods=['GET'])
def testimonial():
    return render_template('testimonial.html')

@app.route('/treatment', methods=['GET'])
def treatment():
    return render_template('treatment.html')

# Route for the prediction page
@app.route('/predict', methods=['POST'])
def upload():
    try:
        # Check if 'file' exists in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        f = request.files['file']

        # Check if the file is selected
        if f.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the file
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)  # Create uploads folder if it doesn't exist

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)

        # Extract prediction result
        # prediction = preds[0][0]  # Assuming binary classification
        # if prediction > 0.7:
        #     result = f'Negative for COVID-19 (Confidence: {prediction:.2f})'
        # else:
        #     result = f'Positive for COVID-19 (Confidence: {1 - prediction:.2f})'

        # Return the prediction result
        print(result)
        return jsonify({'prediction':result})
    

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



