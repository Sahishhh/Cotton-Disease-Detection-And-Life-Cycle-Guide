import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load model and labels
model_path = os.path.join(os.path.dirname(__file__), 'cotton_mobilenetv2_model.h5')
labels_path = os.path.join(os.path.dirname(__file__), 'class_indices.json')

try:
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, 'r') as f:
        class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}
except Exception as e:
    print(json.dumps({'error': str(e)}))
    sys.exit(1)

# Get image path from Node.js
if len(sys.argv) < 2:
    print(json.dumps({'error': 'No image path provided'}))
    sys.exit(1)

img_path = sys.argv[1]
try:
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    result = {'prediction': predicted_class, 'confidence': round(confidence, 2)}
    print(json.dumps(result))

except Exception as e:
    print(json.dumps({'error': str(e)}))
