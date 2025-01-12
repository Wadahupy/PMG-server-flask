from flask import Flask, request, jsonify
import pandas as pd
from datetime import timedelta, datetime
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model
model = joblib.load('RandomForestRegressor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()
        if 'deathDate' not in data:
            return jsonify({"error": "Missing 'deathDate' in the request data"}), 400
        
        death_date_input = data['deathDate']
        death_date = pd.to_datetime(death_date_input)
        today = pd.Timestamp(datetime.now().date())
        
        # Validate death date
        if death_date > today:
            return jsonify({"error": "Pet death date cannot be in the future."}), 400

        # Prepare test features (ensure consistency with training data)
        test_features = pd.DataFrame({
            'Death Month': [death_date.month],
            'Burial Weekday': [0],  # Dummy placeholder
            'Burial Month': [death_date.month]  # Default assumption
        })

        # Predict days between
        predicted_days_between = model.predict(test_features)[0]

        # Recommend burial date
        recommended_burial_date = death_date + timedelta(days=round(predicted_days_between))

        # Get the weekday of the recommended burial date
        burial_weekday = recommended_burial_date.day_name()

        # Format dates to Y-M-D (YYYY-MM-DD)
        formatted_death_date = death_date.strftime("%Y-%m-%d")
        formatted_burial_date = recommended_burial_date.strftime("%Y-%m-%d")

        # Return JSON response
        return jsonify({
            "deathDate": formatted_death_date,
            "predictedDaysBetween": round(predicted_days_between),
            "burialDate": formatted_burial_date,
            "burialWeekday": burial_weekday
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
