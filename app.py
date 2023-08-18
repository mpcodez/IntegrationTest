from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
from flask import send_from_directory

# ... (other imports)

app = Flask(__name__)

# Load the pre-trained model
model = load_model('MNIST.h5')
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]  # Replace with your class names

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['image']

    if uploaded_file.filename != '':
        image_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(image_path)
        return classify_image(image_path)
    else:
        return jsonify({'error': 'No file uploaded'})

@app.route('/classify', methods=['POST'])

def classify():
    uploaded_file = request.files['image']

    if uploaded_file.filename != '':
        image_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(image_path)
        return classify_image(image_path)
    else:
        return jsonify({'error': 'No file uploaded'})

def classify_image(image_path):

    # load the image
    image = Image.open(image_path)
    image = np.reshape(image, [-1, 784])
    image = image.astype('float32') / 255
    
    # make predictions
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]

    with open('readme.txt', 'a') as f:
        f.write("Image Path: " + image_path + "; Predicted Class: " + predicted_class + "\n")

    return jsonify({'prediction': predicted_class, 'image_url': image_path})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)