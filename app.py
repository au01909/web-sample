from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from pose_processor import PoseProcessor
import tensorflow as tf
import h5py
import json

app = Flask(__name__)
CORS(app)

# Load LSTM Model
def load_lstm_model():
    custom_objects = {
        'Orthogonal': tf.keras.initializers.Orthogonal,
        'Sequential': tf.keras.models.Sequential
    }

    with h5py.File("lstm-trained.h5", 'r') as f:
        model_config = json.loads(f.attrs['model_config'])
        model = tf.keras.models.model_from_json(json.dumps(model_config), custom_objects=custom_objects)
        
        for layer in model.layers:
            if layer.name in f['model_weights']:
                layer.set_weights([f['model_weights'][layer.name][name] 
                                 for name in f['model_weights'][layer.name].attrs['weight_names']])
    return model

model = load_lstm_model()
pose_processor = PoseProcessor()
frame_buffer = []

@app.route('/predict', methods=['POST'])
def predict():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    frame = cv2.imdecode(np.frombuffer(request.files['frame'].read(), np.uint8), cv2.IMREAD_COLOR)
    landmarks = pose_processor.process_frame(frame)
    
    if landmarks is None:
        return jsonify({'prediction': 'neutral', 'confidence': 0.0})
    
    frame_buffer.append(landmarks)
    
    if len(frame_buffer) == 20:
        prediction = model.predict(np.expand_dims(frame_buffer, axis=0))[0][0]
        frame_buffer.clear()
        return jsonify({
            'prediction': 'violent' if prediction > 0.5 else 'neutral',
            'confidence': float(prediction)
        })
    
    return jsonify({'status': 'collecting', 'count': len(frame_buffer)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
