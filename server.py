import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allow the webpage to talk to this server

# Load the model ONCE when the server starts
print("Loading Keras model...")
model = tf.keras.models.load_model('mnist_cnn_classifier.h5')
print("Model loaded!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the pixel data from the webpage
        data = request.json
        pixels = data['pixels'] # Expecting a flat list or 28x28 list
        
        # 2. Convert to NumPy array and reshape for the model
        # The model expects (1, 28, 28, 1)
        # We assume the webpage sends us values 0-1 (normalized)
        input_data = np.array(pixels).reshape(1, 28, 28, 1)
        
        # 3. Make Prediction
        prediction = model.predict(input_data)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)
        
        # 4. Send result back to webpage
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)