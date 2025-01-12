# Pet Burial Date Prediction
This project is a Flask-based web application that predicts the most suitable burial date for a pet based on the provided death date. It leverages a trained machine learning model (Random Forest Regressor) to make predictions.

## Features
  - Input pet's death date to receive a recommended burial date.
  - Automatic prevention of future death date selection.
  - Displays additional details such as the number of predicted days between death and burial and the burial weekday.

## Prerequisites
Ensure you have the following installed:
  - Python 3.8+
  - Flask
  - pandas
  - scikit-learn
  - joblib

## File Structure
  - app.py: Main application logic.
  - index.html: Frontend template.
  - RandomForestRegressor.pkl: Trained machine learning model.
  - processed_data.csv: Data used for model training (if applicable).
  - requirements.txt: Python dependencies.
