from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import webbrowser

app = Flask(__name__)

model = tf.keras.models.load_model(r"C:\Users\sarth\Downloads\PBL\alzheimers_cnn_model_with_class_weights_and_early_stopping.h5")
CATEGORIES = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']
IMG_SIZE = 64

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file).convert('L').resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.stack([img_array]*3, axis=-1)

        prediction = model.predict(img_array)
        predicted_class = CATEGORIES[np.argmax(prediction)]

        return render_template('index.html', prediction=predicted_class)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    port = 5000  # Set the port
    url = f"http://127.0.0.1:{port}"

    print(f"Server running! Click here to open: {url}")  # Clickable link in terminal
    webbrowser.open(url)  # Auto-opens in the default browser

    app.run(debug=True, port=port)  # Start the Flask app
    app.run(debug=True)
