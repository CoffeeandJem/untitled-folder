# app.py

import flask
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
import joblib
import pandas as pd
import numpy as np
import os

# Import the preprocessing function we created
try:
    from preprocessing import preprocess_for_prediction
    print("Preprocessing function loaded successfully.")
except ImportError:
    print("Error: Could not import 'preprocess_for_prediction' from preprocessing.py.")
    # Define a dummy function or exit if preprocessing is critical
    def preprocess_for_prediction(input_data, region_encoder):
        print("ERROR: PREPROCESSING FUNCTION IS MISSING!")
        return None

# --- Configuration ---
# Define paths to artifacts relative to app.py
ARTIFACTS_DIR = 'artifacts'
XGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'xgb_career_model.pkl')
SVM_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'svm_career_model.pkl') # Path if you want to load SVM too
REGION_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'region_encoder.pkl')
LABEL_ENCODER_Y_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl')

# --- Load Artifacts on Startup ---
print("Loading artifacts...")
try:
    # Load the XGBoost pipeline (includes scaler + model)
    XGB_PIPELINE = joblib.load(XGB_MODEL_PATH)
    print(f"XGBoost model pipeline loaded from {XGB_MODEL_PATH}")

    # Load the fitted region encoder
    REGION_ENCODER = joblib.load(REGION_ENCODER_PATH)
    print(f"Region encoder loaded from {REGION_ENCODER_PATH}")
    print(f"  - Known region categories (sample): {list(REGION_ENCODER.classes_[:5])}...")


    # Load the fitted target label encoder
    LABEL_ENCODER_Y = joblib.load(LABEL_ENCODER_Y_PATH)
    print(f"Target label encoder loaded from {LABEL_ENCODER_Y_PATH}")
    print(f"  - Known target categories (sample): {list(LABEL_ENCODER_Y.classes_[:5])}...")

    artifacts_loaded = True
except FileNotFoundError as e:
    print(f"Error loading artifacts: {e}")
    print("Please ensure all .pkl files (xgb_career_model.pkl, region_encoder.pkl, label_encoder.pkl)")
    print(f"are present in the '{ARTIFACTS_DIR}' directory relative to app.py.")
    artifacts_loaded = False
except Exception as e:
    print(f"An unexpected error occurred during artifact loading: {e}")
    artifacts_loaded = False

# --- Create Flask App ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing requests from your Next.js app

# --- Define API Endpoints ---

@app.route('/')
def home():
    # Simple route to check if the API is running
    return "Career Recommendation API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to predict career based on input JSON data.
    """
    if not artifacts_loaded:
        return jsonify({"error": "Server configuration error: Artifacts not loaded."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    print(f"\nReceived data for prediction: {data}") # Log received data

    # Basic input validation (add more specific checks as needed)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON data received"}), 400

    # Add checks for essential keys if necessary, e.g.:
    # required_keys = ['about', 'region', 'following', 'recommendations_count', 'languages', 'education']
    # if not all(key in data for key in required_keys):
    #     return jsonify({"error": f"Missing required input fields. Need: {required_keys}"}), 400

    try:
        # 1. Preprocess the input data using the loaded region encoder
        print("Preprocessing input data...")
        X_processed = preprocess_for_prediction(data, REGION_ENCODER)

        if X_processed is None:
             print("Preprocessing failed.")
             return jsonify({"error": "Failed to preprocess input data."}), 500
        
        print(f"Processed features shape: {X_processed.shape}")
        # print(f"Processed features data:\n{X_processed.head()}") # Optional: log processed data

        # 2. Make prediction using the loaded XGBoost pipeline
        print("Making prediction...")
        # Predict expects a 2D array-like structure
        prediction_encoded = XGB_PIPELINE.predict(X_processed)
        print(f"Encoded prediction: {prediction_encoded}")

        # 3. Decode the prediction using the loaded target encoder
        # inverse_transform expects a 1D array-like structure
        predicted_label = LABEL_ENCODER_Y.inverse_transform(prediction_encoded)
        recommendation = predicted_label[0] # Get the string label
        print(f"Decoded recommendation: {recommendation}")

        # 4. Return the result
        return jsonify({'recommendation': recommendation})

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Log the full traceback for debugging if needed
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Use port 5000 (or another port if 5000 is busy)
    # host='0.0.0.0' makes it accessible on your network (useful for testing from other devices)
    # debug=True provides auto-reloading and more detailed errors during development
    app.run(debug=True, host='0.0.0.0', port=5001)