import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np

class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == 'Linear Regression':
            return LinearRegression()
        elif model_name == 'Support Vector Machine':
            return SVR()
        elif model_name == 'Random Forest':
            return RandomForestRegressor()
        elif model_name == 'Gradient Boosting Regressor':
            return GradientBoostingRegressor()
        elif model_name == 'XGBRegressor':
            return XGBRegressor()
        else:
            raise ValueError(f"Model '{model_name}' not recognized!")

def load_data(file_name):
    return pd.read_excel(file_name)

def preprocess_data(data):
# Check for missing values
    if data.isnull().any().any():
        data.dropna(inplace=True)
        #raise ValueError("The data contains missing values. Please ensure the data is cleaned before processing.")

#'Gender','Age','Occupation','City_Category','Marital_Status'

    #changes data from F/M to 0/1 meaning it is easyer for computer understand

    GEN_MAP = {'F': 0 ,'M': 1}
    AG_MAP = {
    '0-17': 0,
    '18-25': 1,
    '26-35': 2,
    '36-45': 3,
    '46-50': 4,
    '51-55': 5,
    '55+': 6}
    CC_MAP = {
        'A': 1,
        'B': 2,
        'C': 3
	}
    

    data['Age'].replace(AG_MAP,inplace=True)
    data['Gender'].replace(GEN_MAP,inplace=True)
    data['City_Category'].replace(CC_MAP,inplace=True)
	
    X = data.drop(['User_ID', 'Product_ID', 'Stay_In_Current_City_Years', 'Purchase','Product_Category_1','Product_Category_2','Product_Category_3'], axis=1)
    Y = data['Purchase']
    
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    
    return X_scaled, y_scaled, sc, sc1

def split_data(X_scaled, y_scaled):
    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    model_names = [
        'Linear Regression',
        'Support Vector Machine',
        'Random Forest',
        'Gradient Boosting Regressor',
        'XGBRegressor'
    ]
    
    models = {}
    for name in model_names:
        #Display model name
        print(f"Training model: {name}")
        model = ModelFactory.get_model(name)
        model.fit(X_train, y_train.ravel())
        models[name] = model
        #display when model trained successfully
        print(f"{name} trained successfully.")
        
    return models


def evaluate_models(models, X_test, y_test):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
        
    return rmse_values

def plot_model_performance(rmse_values):
    plt.figure(figsize=(10,7))
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    bars = plt.bar(models, rmse, color=['blue', 'green', 'red', 'purple', 'orange'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_best_model(models, rmse_values):
    best_model_name = min(rmse_values, key=rmse_values.get) 
    best_model = models[best_model_name]
    dump(best_model, "Train.joblib")

def predict_new_data(loaded_model, sc, sc1):
    X_test1 = sc.transform(np.array([[0,42,62812.09301,11609.38091,238961.2505]]))
    pred_value = loaded_model.predict(X_test1)
    print(pred_value)
    
    # Ensure pred_value is a 2D array before inverse transform
    if len(pred_value.shape) == 1:
        pred_value = pred_value.reshape(-1, 1)

    print("Predicted output: ", sc1.inverse_transform(pred_value))

if __name__ == "__main__":
    try: #add try except to handle missing value error
        data = load_data('train.xlsx')
        X_scaled, y_scaled, sc, sc1 = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
        models = train_models(X_train, y_train)
        rmse_values = evaluate_models(models, X_test, y_test)
        plot_model_performance(rmse_values)
        save_best_model(models, rmse_values)
        loaded_model = load("Train.joblib")
        predict_new_data(loaded_model, sc, sc1)
    except ValueError as ve:
        print(f"Error: {ve}")

