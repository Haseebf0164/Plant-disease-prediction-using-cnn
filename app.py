from flask import Flask, render_template, request, flash, redirect
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import logging
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['SECRET_KEY'] = os.urandom(24)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Load model
try:
    model = tf.keras.models.load_model('plant_disease_detection_using_cnn.keras')
    logger.info(f"Model loaded successfully with output shape: {model.output_shape}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Optional: Save class names once
# with open('class_names.pkl', 'wb') as f:
#     pickle.dump(class_names, f)

# Validate model output shape
if model.output_shape[-1] != len(class_names):
    raise ValueError(f"Mismatch: model expects {model.output_shape[-1]} outputs, but {len(class_names)} class names provided.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(256, 256)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

def predict(img_path):
    try:
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        return class_names[class_idx], confidence
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Unknown", 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Only png, jpg, jpeg allowed.', 'error')
            return redirect(request.url)

        try:
            # Save uploaded file
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict
            predicted_class, confidence = predict(filepath)

            # Cleanup old files (keep only latest 100)
            files = sorted(Path(app.config['UPLOAD_FOLDER']).glob('*'), key=os.path.getmtime)
            for old_file in files[:-100]:
                try:
                    old_file.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting old file: {old_file} - {e}")

            return render_template(
                'index.html',
                prediction=predicted_class,
                confidence=f"{confidence:.2%}",
                image_path=filepath
            )
        except Exception as e:
            logger.error(f"Upload or prediction failed: {e}")
            flash('Something went wrong while processing the image.', 'error')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
