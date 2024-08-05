from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

model = joblib.load('pkls/model.pkl')
with open('pkls/X_train_columns.pkl', 'rb') as file:
    X_train_columns = pickle.load(file)

# Load the label encoders
with open('pkls/label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

with open('pkls/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def encode_labels(data, label_encoders):
    for column, encoder in label_encoders.items():
        data[column] = encoder.transform(data[column].astype(str))
    return data

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/home')
def home_page():
    return render_template('appraiseme.html')

@app.route('/value', methods=['POST'])
def home():
    car_make = request.form.get('Make')
    car_model = request.form.get('Model')
    car_year = int(request.form.get('Year'))
    car_miles = int(request.form.get('Miles'))

    print(car_model)
    print(car_make)
    print(car_year)
    print(car_miles)

    input_data = pd.DataFrame({
    'Year': [car_year],
    'Miles': [car_miles],
    'Make': [car_make],
    'Model': [car_model]
    })

    input_data = encode_labels(input_data, label_encoders)

    input_data = input_data[X_train_columns]

    input_data_scaled = scaler.transform(input_data)

    predicted_value = model.predict(input_data_scaled)

    predicted_value = round(predicted_value[0], 2)

    formatted_value = "{:,}".format(predicted_value)

    print("Predicted value:", formatted_value)
    return render_template('after.html', data=formatted_value)

if __name__ == "__main__":
    app.run(debug=True)