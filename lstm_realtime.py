import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def load_model():
    """Load trained LSTM model"""
    # Choose one of these loading methods based on your saved format
    model = tf.keras.models.load_model('model/lstm_model.h5')
    # model = joblib.load('model/lstm_model.pkl') 
    return model

def preprocess_frame(frame):
    """Preprocess frame for LSTM input"""
    # Your preprocessing steps here (resize, normalize, etc.)
    resized = cv2.resize(frame, (224, 224))
    normalized = resized / 255.0
    return normalized
