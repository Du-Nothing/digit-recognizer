from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)
model = tf.keras.models.load_model('digit_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert('L').resize((28, 28))
    img = np.array(img)
    img = 255 - img  # invert colors
    img = img / 255.0
    img = img.reshape(1, 28, 28)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    return f"Predicted Digit: {digit}"

if __name__ == '__main__':
    app.run(debug=True)
