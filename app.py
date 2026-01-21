"""
House Price Prediction - Flask Web Application
===============================================
This Flask app provides a web interface for predicting house prices
using the trained Linear Regression model.

Author: [Your Name]
Matric No: [Your Matric Number]
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ============================================
# LOAD TRAINED MODEL AND ENCODER
# ============================================
print("Loading model and encoder...")

try:
    # Load the trained model
    with open('model/house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully!")
    
    # Load the label encoder for Neighborhood
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Label encoder loaded successfully!")
    
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Make sure 'house_price_model.pkl' and 'label_encoder.pkl' are in the /model/ directory")
    exit()

# Get list of valid neighborhoods
valid_neighborhoods = label_encoder.classes_.tolist()

# ============================================
# HOME PAGE ROUTE
# ============================================
@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html', neighborhoods=valid_neighborhoods)

# ============================================
# PREDICTION ROUTE
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return prediction"""
    
    try:
        # Get form data
        overall_qual = int(request.form['overall_qual'])
        gr_liv_area = float(request.form['gr_liv_area'])
        total_bsmt_sf = float(request.form['total_bsmt_sf'])
        garage_cars = int(request.form['garage_cars'])
        year_built = int(request.form['year_built'])
        neighborhood = request.form['neighborhood']
        
        # Encode neighborhood
        neighborhood_encoded = label_encoder.transform([neighborhood])[0]
        
        # Prepare features in correct order
        # Order: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt, Neighborhood_Encoded
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            neighborhood_encoded
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Format prediction as currency
        predicted_price = f"${prediction:,.2f}"
        
        return render_template(
            'index.html',
            neighborhoods=valid_neighborhoods,
            prediction=predicted_price,
            show_result=True,
            # Return input values to display them
            input_values={
                'overall_qual': overall_qual,
                'gr_liv_area': gr_liv_area,
                'total_bsmt_sf': total_bsmt_sf,
                'garage_cars': garage_cars,
                'year_built': year_built,
                'neighborhood': neighborhood
            }
        )
        
    except ValueError as e:
        error_message = f"Invalid input: {str(e)}"
        return render_template(
            'index.html',
            neighborhoods=valid_neighborhoods,
            error=error_message
        )
    
    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        return render_template(
            'index.html',
            neighborhoods=valid_neighborhoods,
            error=error_message
        )

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
    