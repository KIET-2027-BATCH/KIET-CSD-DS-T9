# Import necessary libraries
from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model and scaler
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = []
    
    try:
        # Extract all features from the form
        features.append(float(request.form['age']))
        features.append(int(request.form['sex']))
        features.append(int(request.form['cp']))
        features.append(float(request.form['trestbps']))
        features.append(float(request.form['chol']))
        features.append(int(request.form['fbs']))
        features.append(int(request.form['restecg']))
        features.append(float(request.form['thalach']))
        features.append(int(request.form['exang']))
        features.append(float(request.form['oldpeak']))
        features.append(int(request.form['slope']))
        features.append(int(request.form['ca']))
        features.append(int(request.form['thal']))
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        # Prepare result
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1])
        }
        
        # Return prediction
        return render_template('index.html', result=result)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)