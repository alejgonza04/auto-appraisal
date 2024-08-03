import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle as pickle
import joblib


data = pd.read_csv('carvana.csv')
data2 = pd.read_csv('carvana_car_sold-2022-08.csv')
data3 = pd.read_csv('car_web_scraped_dataset.csv')

print(data3.columns)

data[['Make', 'Model']] = data['Name'].str.split(n=1, expand=True)
data3[['Make', 'Model']] = data3['name'].str.split(n=1, expand=True) 

data.drop('Name', axis=1, inplace=True)
data3.drop('name', axis=1, inplace=True)

data.shape

cols_drop = ['vehicle_id', 'stock_number', 'trim', 'discounted_sold_price', 'partnered_dealership',
'delivery_fee', 'earliest_delivery_date', 'sold_date']

cols_drop2 = ['color', 'condition']

data2 = data2.drop(columns=cols_drop)
data3 = data3.drop(columns=cols_drop2)

data2.shape

data2.rename(columns={
    'year' : 'Year',
    'make' : 'Make',
    'model' : 'Model',
    'miles' : 'Miles',
    'sold_price' : 'Sold_Price'
}, inplace=True)

data3.rename(columns={
    'year' : 'Year',
    'Make' : 'Make',
    'Model' : 'Model',
    'miles' : 'Miles',
    'price' : 'sale_price'
}, inplace=True)

def clean_price(price):
    if isinstance(price, str):
        # Remove currency symbols and commas
        price = price.replace('$', '').replace(',', '')
    return pd.to_numeric(price, errors='coerce')

def clean_miles(miles):
    if isinstance(miles, str):
        miles = miles.replace(',', '').replace(' miles', '')
    return pd.to_numeric(miles, errors='coerce')

data3['sale_price'] = data3['sale_price'].apply(clean_price)
data3['Miles'] = data3['Miles'].apply(clean_miles)

# Ensure 'Miles' columns are of the same type
data['Miles'] = pd.to_numeric(data['Miles'], errors='coerce')
data2['Miles'] = pd.to_numeric(data2['Miles'], errors='coerce')
data3['Miles'] = pd.to_numeric(data3['Miles'], errors='coerce')

combined_data = pd.merge(pd.merge(data, data2, on=['Year', 'Make', 'Model', 'Miles'], how='outer'), 
                         data3, on=['Year', 'Make', 'Model', 'Miles'], how='outer')

print(combined_data.columns)

print(combined_data.head())

combined_data['Price'] = combined_data['Price'].fillna(combined_data['sale_price'])
combined_data = combined_data.drop(columns=['sale_price'])
combined_data['Price'] = combined_data['Price'].fillna(combined_data['Sold_Price'])

combined_data = combined_data.drop(columns=['Sold_Price'])

print(combined_data.dtypes)

combined_data.shape

combined_data = combined_data.dropna()

combined_data.isnull().sum()

min_year = combined_data['Year'].min()

print(f"The smallest value for the 'Year' column is: {min_year}")

print(combined_data.shape)

non_numeric_columns = ['Make', 'Model']

label_encoders = {}
for column in non_numeric_columns:
  le = LabelEncoder()
  combined_data[column] = le.fit_transform(combined_data[column].astype(str))
  label_encoders[column] = le

X = combined_data.drop('Price', axis=1)
Y = combined_data['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def model_evaluation(true, predicted):
  rmse = np.sqrt(mean_squared_error(true, predicted))
  mse = mean_squared_error(true, predicted)
  mae = mean_absolute_error(true, predicted)
  r2_square = r2_score(true, predicted)
  return mae, rmse, r2_square

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
'''
# define models and parameters for GridSearchCV
models = {
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor()
}

param_grids = {
    "Random Forest Regressor": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "XGBRegressor": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 10]
    }
}

# evaluate models with hyperparameter tuning
for model_name, model in models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, Y_train)

    best_model = grid_search.best_estimator_

    Y_train_p = best_model.predict(X_train_scaled)
    Y_test_p = best_model.predict(X_test_scaled)

    model_train_mae, model_train_rmse, model_train_r2 = model_evaluation(Y_train, Y_train_p)
    model_test_mae, model_test_rmse, model_test_r2 = model_evaluation(Y_test, Y_test_p)

    print(model_name)
    print("Best Parameters:", grid_search.best_params_)
    print('Model Training Performances')
    print(f"R2 Score: {model_train_r2}")
    print(f"RMSE: {model_train_rmse}")
    print(f"MAE: {model_train_mae}")

    print('Model Test Performances')
    print(f"R2 Score: {model_test_r2}")
    print(f"RMSE: {model_test_rmse}")
    print(f"MAE: {model_test_mae}")

    print('\n')'''
'''Random Forest Regressor
Best Parameters: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}
Model Training Performances
R2 Score: 0.9684374842023856
RMSE: 1719.7076127418825
MAE: 884.4977945499543
Model Test Performances
R2 Score: 0.883680742111431
RMSE: 3322.085406383798
MAE: 1632.4166030053875

Best Parameters: {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 200}
Model Training Performances
R2 Score: 0.9625868544478987
RMSE: 1911.7171642600324
MAE: 965.7406287192564
Model Test Performances
R2 Score: 0.8626296243136744
RMSE: 3680.6652577555533
MAE: 1844.0141156088052


XGBRegressor
Best Parameters: {'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 200}
Model Training Performances
R2 Score: 0.9078741423481613
RMSE: 2938.052685611779
MAE: 1689.3716020208847
Model Test Performances
R2 Score: 0.8713853361955057
RMSE: 3493.254485933821
MAE: 1867.341445995628


model2 = {
    'LinearRegression': LinearRegression()
}

param_grids = {
    'LinearRegression': {}
}

for model_name, model in model2.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, Y_train)

    best_model = grid_search.best_estimator_

    Y_train_p = best_model.predict(X_train_scaled)
    Y_test_p = best_model.predict(X_test_scaled)

    model_train_mae, model_train_rmse, model_train_r2 = model_evaluation(Y_train, Y_train_p)
    model_test_mae, model_test_rmse, model_test_r2 = model_evaluation(Y_test, Y_test_p)

    print(model_name)
    print("Best Parameters:", grid_search.best_params_)
    print('Model Training Performances')
    print(f"R2 Score: {model_train_r2}")
    print(f"RMSE: {model_train_rmse}")
    print(f"MAE: {model_train_mae}")

    print('Model Test Performances')
    print(f"R2 Score: {model_test_r2}")
    print(f"RMSE: {model_test_rmse}")
    print(f"MAE: {model_test_mae}")
    LinearRegression
        Best Parameters: {}
        Model Training Performances
        R2 Score: 0.27258084150644324
        RMSE: 8255.838237893306
        MAE: 5657.34540772094
        Model Test Performances
        R2 Score: 0.28454724660814756
        RMSE: 8239.018190400164
        MAE: 5611.402309037089'''

test_data = pd.DataFrame({
    'Year': [2009],
    'Miles': [120000],
    'Make': ['Honda'],
    'Model': ['Accord']
})

def encode_labels(data, label_encoders):
    for column, encoder in label_encoders.items():
        # Ensure that all labels in data are known to the encoder
        data[column] = encoder.transform(data[column].astype(str))
    return data

test_data = encode_labels(test_data, label_encoders)

test_data_scaled = scaler.transform(test_data)

best_model = RandomForestRegressor(max_depth=30, min_samples_split=5, n_estimators=200,
                                   random_state=42)
best_model.fit(X_train_scaled, Y_train)

Y_train_p = best_model.predict(X_train_scaled)
Y_test_p = best_model.predict(X_test_scaled)

model_train_mae, model_train_rmse, model_train_r2 = model_evaluation(Y_train, Y_train_p)
model_test_mae, model_test_rmse, model_test_r2 = model_evaluation(Y_test, Y_test_p)

print('Model Training Performances')
print(f"R2 Score: {model_train_r2}")
print(f"RMSE: {model_train_rmse}")
print(f"MAE: {model_train_mae}")

print('Model Test Performances')
print(f"R2 Score: {model_test_r2}")
print(f"RMSE: {model_test_rmse}")
print(f"MAE: {model_test_mae}")

performance_metrics = {
    'model_train_r2': model_train_r2,
    'model_train_rmse': model_train_rmse,
    'model_train_mae': model_train_mae,
    'model_test_r2': model_test_r2,
    'model_test_rmse': model_test_rmse,
    'model_test_mae': model_test_mae
}

predicted_price = best_model.predict(test_data_scaled)
print(f"Predicted Price: ${predicted_price[0]:.2f}")

joblib.dump(best_model, 'model.pkl', compress=3, protocol=4)


with open('X_train_columns.pkl', 'wb') as file:
    pickle.dump(X_train.columns.tolist(), file)

with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

with open('performance_metrics.pkl', 'wb') as file:
    pickle.dump(performance_metrics, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

