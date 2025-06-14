import pandas as pd
from prophet import Prophet
import joblib
from flask import Flask, request, jsonify
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Model ---
# Define the path to your saved model relative to app.py
# IMPORTANT: Ensure 'prophet_model.joblib' is in the same directory as app.py
MODEL_PATH = 'prophet_model.joblib'

# Check if the model file exists before trying to load it
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}. "
          "Please ensure main.py has been run to generate it and it's in the correct location.")
    # In a production scenario, you might want to log this and handle it more gracefully.
    # For this challenge, exiting is acceptable if the model is critical.
    exit(1) # Exit if model is critical and not found

# Load the trained model when the Flask app starts
try:
    model = joblib.load(MODEL_PATH)
    print(f"Prophet model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1) # Exit if model loading fails

# --- Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request (JSON format)
        # force=True allows parsing even if Content-Type header isn't strictly application/json
        data = request.get_json(force=True)
        print(f"Received prediction request with data: {data}")

        # Extract year and month from the request
        year = data['year']
        month = data['month']

        # Basic input validation
        if not isinstance(year, int) or not isinstance(month, int) or not (1 <= month <= 12):
            return jsonify({'error': 'Invalid year or month. Year must be an integer, month must be an integer between 1 and 12.'}), 400

        # Create a DataFrame for the prediction date
        # Prophet requires a 'ds' column (datetime)
        # Pad month with a leading zero if it's a single digit (e.g., 1 -> 01)
        prediction_date_str = f"{year}-{str(month).zfill(2)}-01"
        future_df = pd.DataFrame({'ds': [pd.to_datetime(prediction_date_str)]})

        # Make prediction using the loaded Prophet model
        forecast = model.predict(future_df)

        # Extract the predicted value (yhat)
        predicted_value = float(forecast['yhat'].iloc[0]) # Convert to standard float for JSON

        # Return the prediction as JSON
        return jsonify({'prediction': predicted_value})

    except KeyError as e:
        # Handles cases where 'year' or 'month' might be missing in the request JSON
        return jsonify({'error': f'Missing key in request data: {e}. Expected "year" and "month".'}), 400
    except Exception as e:
        # Catch any other unexpected errors during the prediction process
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

# --- Basic Health Check Endpoint (Optional but good practice) ---
# This endpoint responds to GET requests at the root '/' or '/health'
# It's useful to check if your deployed service is running and accessible.
@app.route('/health', methods=['GET'])
def health_check():
    # You can add more checks here, e.g., if the model is actually loaded
    return jsonify({'status': 'healthy', 'model_loaded': True}), 200

# --- Run the Flask App (for local development/testing) ---
if __name__ == '__main__':
    # When you run 'python app.py' directly, this block executes.
    # debug=True allows for automatic reloading on code changes and provides more detailed error messages.
    # host='0.0.0.0' makes the server accessible from outside your local machine (e.g., a virtual machine)
    # port=5000 is the default Flask port
    app.run(debug=True, host='0.0.0.0', port=5000)