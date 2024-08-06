import sys
import importlib
importlib.reload(sys)

from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model('C:/Users/DELL/OneDrive/Desktop/Project/Vgg16_weight_model.h5')

def preprocess_image(image):
    image = image.resize((120, 120))
    image = np.array(image)
    image = image / 255.0
    return image

def predict_class(image_path):
    image = Image.open(image_path)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    return predicted_class

def get_defect_name(class_number):
    defect_names = {
        1: "Pitted Surface Defect",
        2: "Crazing Defect",
        3: "Scratches Defect",
        4: "Patches Defect"
    }
    return defect_names.get(class_number, "Unknown defect")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_class = predict_class(filepath)
        defect_name = get_defect_name(predicted_class)
        os.remove(filepath)  # Remove the uploaded file after prediction
        return render_template('index.html', prediction=predicted_class, defect_name=defect_name)

if __name__ == '__main__':
    app.run(debug=True)
