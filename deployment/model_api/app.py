from flask import Flask, request, jsonify
import joblib
import pandas as pd
import math

EXPECTED_FEATURES = [
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "location_name_Bank/Savings and Loan",
    "location_name_Church/Synagogue/Temple/Mosque",
    "location_name_Commercial/Office Building",
    "location_name_Construction Site",
    "location_name_Department/Discount Store",
    "location_name_Drug Store/Doctor's Office/Hospital",
    "location_name_Highway/Road/Alley/Street/Sidewalk",
    "location_name_Hotel/Motel/Etc.",
    "location_name_Industrial Site",
    "location_name_Other/Unknown",
    "location_name_Parking/Drop Lot/Garage",
    "location_name_Residence/Home",
    "location_name_Restaurant",
    "location_name_School-Elementary/Secondary",
    "crime_against_Person",
    "crime_against_Property",
    "crime_against_Society",
    "offense_category_name_Assault Offenses",
    "offense_category_name_Burglary/Breaking & Entering",
    "offense_category_name_Destruction/Damage/Vandalism of Property",
    "offense_category_name_Drug/Narcotic Offenses",
    "offense_category_name_Fraud Offenses",
    "offense_category_name_Gambling Offenses",
    "offense_category_name_Homicide Offenses",
    "offense_category_name_Human Trafficking",
    "offense_category_name_Kidnapping/Abduction",
    "offense_category_name_Larceny/Theft Offenses",
    "offense_category_name_Motor Vehicle Theft",
    "offense_category_name_Other/Unknown",
    "offense_category_name_Pornography/Obscene Material",
    "offense_category_name_Prostitution Offenses",
    "offense_category_name_Robbery",
    "offense_category_name_Sex Offenses",
    "offense_category_name_Sex Offenses, Non-forcible",
    "offense_category_name_Weapon Law Violations"
]

def preprocess_human_input(raw_data):
    processed_data = {feat: 0.0 for feat in EXPECTED_FEATURES}

    h = raw_data.get("hour", 0)
    m = raw_data.get("month", 1)
    w = raw_data.get("weekday", 0)
    
    processed_data["is_holiday"] = float(raw_data.get("is_holiday", 0))
    processed_data["is_weekend"] = 1.0 if w in [5, 6] else 0.0

    # Cyclical Encoding
    processed_data["hour_sin"] = math.sin(2 * math.pi * h / 24.0)
    processed_data["hour_cos"] = math.cos(2 * math.pi * h / 24.0)
    processed_data["month_sin"] = math.sin(2 * math.pi * m / 12.0)
    processed_data["month_cos"] = math.cos(2 * math.pi * m / 12.0)
    processed_data["weekday_sin"] = math.sin(2 * math.pi * w / 7.0)
    processed_data["weekday_cos"] = math.cos(2 * math.pi * w / 7.0)

    # One-Hot Encoding
    loc_col = f"location_name_{raw_data.get('location_name', '')}"
    if loc_col in processed_data:
        processed_data[loc_col] = 1.0

    crime_against_col = f"crime_against_{raw_data.get('crime_against', '')}"
    if crime_against_col in processed_data:
        processed_data[crime_against_col] = 1.0

    offense_col = f"offense_category_name_{raw_data.get('offense_category_name', '')}"
    if offense_col in processed_data:
        processed_data[offense_col] = 1.0

    return processed_data

app = Flask(__name__)

# Load all models into a dictionary
MODELS = {
    'lgbm': joblib.load('lgbm_model.pkl'),
    'lr': joblib.load('lr_model.pkl')
}

@app.route('/', methods=['GET'])
def home():
    return "IT5006 Phase 3 Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Input validation
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided.'}), 400

        required_fields = ['hour', 'month', 'weekday', 'location_name', 'crime_against', 'offense_category_name']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'success': False, 'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        if not (0 <= int(data['hour']) <= 23):
            return jsonify({'success': False, 'error': 'Invalid hour. Must be between 0 and 23.'}), 400
        if not (1 <= int(data['month']) <= 12):
            return jsonify({'success': False, 'error': 'Invalid month. Must be between 1 and 12.'}), 400
        if not (0 <= int(data['weekday']) <= 6):
            return jsonify({'success': False, 'error': 'Invalid weekday. Must be between 0 and 6.'}), 400

        # Model selection (default to 'lgbm')
        model_type = data.get('model_type', 'lgbm')
        if model_type not in MODELS:
            return jsonify({
                'success': False, 
                'error': f"Invalid model_type. Choose from {list(MODELS.keys())}"
            }), 400

        # Preprocess the input
        processed_features = preprocess_human_input(data)
        df = pd.DataFrame([processed_features])

        # Prediction using selected model
        selected_model = MODELS[model_type]
        prediction = selected_model.predict(df)[0]
        probability = selected_model.predict_proba(df)[0][1]
        
        return jsonify({
            'success': True,
            'model_used': model_type,
            'prediction_label': int(prediction),
            'arrest_probability': float(probability)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)