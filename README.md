<h3 align="center">Auto ApprAIsal</h3>

![aa1](https://github.com/user-attachments/assets/67996769-2c7a-43a8-ae6b-4f315a3dc569)


# Website Link
https://auto-appraisal.onrender.com/

# Technologies Used
Python, Pandas, NumPy, Scikit-Learn, RandomForestRegressor, XGBRegressor, LinearRegression, GridSearchCV, Joblib, Pickle, CSS, HTML, Flask, Django, Render, Google Colab

# Hosting Websites Used
https://render.com/

# About
This project was developed to create a more efficient solution for determining the value of used cars. Inspired by my father's experience as a car salesman, I recognized the need for an application that could automatically appraise vehicles, streamlining the process and providing accurate estimates quickly.

# Features
### Overview
Auto ApprAIsal is a sophisticated web application designed to simplify the process of determining the value of your used car. Leveraging advanced machine learning models trained on extensive datasets, Auto ApprAIsal provides quick and accurate car valuations. Simply provide the make, model, year, and mileage of your vehicle, and our system will deliver an appraisal in seconds.

# How It Works
![aa2](https://github.com/user-attachments/assets/ee2ba21a-f379-4321-b792-aa4a8b837411)

![aa3](https://github.com/user-attachments/assets/b8ab886d-567a-4235-9ad6-d5d9e171bf77)

- I utilized Kaggle web-scraped datasets from platforms like Carvana, using Pandas to manipulate and combine data focused on car make, model, year, and mileage, while dropping irrelevant columns. The data was then cleaned, with non-numeric values encoded using LabelEncoder and scaled with StandardScaler. I trained the model using RandomForestRegressor, XGBRegressor, and LinearRegression, optimizing them with GridSearchCV. RandomForestRegressor achieved the highest R² score and the lowest RMSE and MAE for both training and testing. The models, along with the training columns, encoders, and scalers, were then saved using Pickle and Joblib, and the model was compressed to reduce file size.
- The model was then integrated into a Flask application, creating app routes for different pages. Using Django, HTML, and CSS, I developed a UI that emphasizes conciseness and efficiency. The platform was then deployed using Render.

# Collecting Data To Make a Prediction
![Screenshot 2024-08-03 232537](https://github.com/user-attachments/assets/9a882b9a-49fb-4a3d-bef7-a4806774c49e)

- Users can add information about their car that the machine learning model will use to make a prediction on the car's value.
- Users are given a plethora of car makes and their respective models. 

# The Results
![Screenshot 2024-08-03 230747](https://github.com/user-attachments/assets/8016b158-8528-404f-8e67-3b3c0df1ee64)

- The model predicts the car's value with 86% accuracy, demonstrating a strong alignment between its predictions and the actual values based on the training data.

# Project Use
- 
- 

# Future Development
- 

# Resources
- https://www.kaggle.com/datasets/ayaz11/used-car-price-prediction?select=car_web_scraped_dataset.csv
- https://www.kaggle.com/datasets/ravishah1/carvana-predict-car-prices
- https://www.kaggle.com/datasets/ayaz11/used-car-price-prediction/data

# Author
- Alejandro Gonzalez - https://github.com/alejgonza04
