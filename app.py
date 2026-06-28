from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
from suggestions import DISEASE_SUGGESTIONS, GENERAL_TIPS

app = Flask(__name__)
CORS(app)

# -----------------------------
# LOAD MODEL + ENCODERS
# -----------------------------
print("Loading model and encoders...")

model = joblib.load('model\\model.pkl')
state_encoder = joblib.load('model\\state_encoder.pkl')
month_encoder = joblib.load('model\\month_encoder.pkl')
season_encoder = joblib.load('model\\season_encoder.pkl')
disease_encoder = joblib.load('model\\disease_encoder.pkl')
scaler = joblib.load('model\\scaler.pkl')
features_list = joblib.load('model\\features.pkl')

print("Model loaded successfully!")
print("Diseases:", disease_encoder.classes_)

# -----------------------------
# HELPERS
# -----------------------------
month_to_num = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

def get_season(month_num):
    if month_num in [12, 1, 2]:
        return 'Winter'
    elif month_num in [3, 4, 5]:
        return 'Summer'
    elif month_num in [6, 7, 8, 9]:
        return 'Monsoon'
    else:
        return 'Post-Monsoon'

# -----------------------------
# CLIMATE ESTIMATOR
# -----------------------------
def estimate_climate(state, month):

    if state == "Kerala" and month in ["June", "July", "August", "September"]:
        return {"temp": 28, "humidity": 88, "rainfall": 220, "lai": 3.2, "population": 860}

    elif state == "Rajasthan" and month in ["May", "June"]:
        return {"temp": 42, "humidity": 25, "rainfall": 5, "lai": 0.8, "population": 200}

    elif state == "Delhi" and month in ["November", "December", "January"]:
        return {"temp": 18, "humidity": 60, "rainfall": 10, "lai": 1.5, "population": 11300}

    else:
        return {"temp": 32, "humidity": 60, "rainfall": 50, "lai": 2.0, "population": 500}

# -----------------------------
# ROUTES
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

# -----------------------------
# PREDICTION ROUTE
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        state = data['state']
        month = data['month']

        # -----------------------------
        # VALIDATION
        # -----------------------------
        if state not in state_encoder.classes_:
            return jsonify({'error': 'Unknown state'}), 400

        if month not in month_encoder.classes_:
            return jsonify({'error': 'Unknown month'}), 400

        # -----------------------------
        # CLIMATE VALUES
        # -----------------------------
        climate = estimate_climate(state, month)

        temperature = climate["temp"]
        humidity = climate["humidity"]
        rainfall = climate["rainfall"]
        lai = climate["lai"]
        population_density = climate["population"]

        # -----------------------------
        # ENCODING
        # -----------------------------
        state_encoded = state_encoder.transform([state])[0]
        month_encoded = month_encoder.transform([month])[0]

        month_num = month_to_num[month]
        season = get_season(month_num)
        season_encoded = season_encoder.transform([season])[0]

        # -----------------------------
        # FEATURE ENGINEERING
        # -----------------------------
        temp_humidity_ratio = temperature / max(humidity, 1)
        rainfall_temp = rainfall * temperature
        lai_rainfall = lai * rainfall
        population_temp = population_density * temperature
        temp_squared = temperature ** 2
        rainfall_log = np.log1p(rainfall)
        humidity_squared = humidity ** 2

        is_summer = int(season == 'Summer')
        is_monsoon = int(season == 'Monsoon')
        is_winter = int(season == 'Winter')

        # -----------------------------
        # INPUT DICT
        # -----------------------------
        input_dict = {
            'Temperature_Max (°C)': temperature,
            'Humidity': humidity,
            'Rainfall (mm)': rainfall,
            'LAI': lai,
            'Population_Density': population_density,
            'State_encoded': state_encoded,
            'Month_encoded': month_encoded,
            'Temp_Humidity_Ratio': temp_humidity_ratio,
            'Rainfall_Temp': rainfall_temp,
            'LAI_Rainfall': lai_rainfall,
            'Population_Temp': population_temp,
            'Temp_Squared': temp_squared,
            'Rainfall_Log': rainfall_log,
            'Humidity_Squared': humidity_squared,
            'Is_Summer': is_summer,
            'Is_Monsoon': is_monsoon,
            'Is_Winter': is_winter,
            'Season_encoded': season_encoded
        }

        # -----------------------------
        # FIXED FEATURE ALIGNMENT (IMPORTANT)
        # -----------------------------
        X = pd.DataFrame([input_dict])
        X = X.reindex(columns=features_list)
        X = scaler.transform(X)

        # -----------------------------
        # PREDICTION
        # -----------------------------
        probabilities = model.predict_proba(X)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        predictions = []
        for idx in top_indices:
            disease = disease_encoder.inverse_transform([idx])[0]
            predictions.append({
                'disease': disease,
                'probability': round(float(probabilities[idx]) * 100, 2),
                'suggestions': DISEASE_SUGGESTIONS.get(disease, GENERAL_TIPS)
            })

        confidence = predictions[0]['probability']

        return jsonify({
            'success': True,
            'predictions': predictions,
            'confidence_value': float(confidence),
            'confidence_level': (
                "High" if confidence > 80 else
                "Moderate" if confidence > 60 else
                "Low"
            )
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)