from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
from lstm_realtime import load_model, preprocess_frame

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load model during startup
model = None
def initialize_model():
    global model
    model = load_model()  # Implement this in lstm_realtime.py
    print("Model loaded successfully")

initialize_model()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for violence prediction"""
    try:
        # Get frame from request
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
            
        frame_file = request.files['frame']
        frame = cv2.imdecode(np.frombuffer(frame_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Preprocess frame for LSTM model
        processed_frame = preprocess_frame(frame)
        
        # Make prediction (adapt for your LSTM's sequence requirements)
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))
        
        # Return confidence score
        return jsonify({
            'violence': float(prediction[0][0]),
            'message': 'Success'
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
