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
from joblib import dump
from joblib import load
#**Varibles

filepath = r'D:\ExamTest_11_09\train.xlsx'

#**Tasks W6_D1
#Import the dataset
data = pd.read_excel(filepath)

#** Create the input dataset from the original dataset by dropping the irrelevant features
X= data.drop(['Gender', 'Age', 'Occupation', 'City_Category', 'Marital_Status'],axis=1)

#**Create the output dataset from the original dataset.
#store output variable in Y
Y= data['Purchase']

#**Transform the input dataset into a percentage based weighted value between 0 and 1.
#! Using MinMaxScaler
scaler = MinMaxScaler()
X_Scaled = scaler.fit_transform(X)

#**Transform the output dataset into a percentage based weighted value between 0 and 1
scaler1= MinMaxScaler()
#Reshape
Y_reshape= Y.values.reshape(-1,1)
Y_scaled=scaler1.fit_transform(Y_reshape)

#**Split the dataset into the training set and test set
#! The train_test_split() method is used to split our data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_Scaled, Y_scaled, test_size=0.2, random_state=42)

#**Import and Initialize the Models
lr = LinearRegression() 
svm = SVR() 
rf = RandomForestRegressor() 
gbr = GradientBoostingRegressor() 
xg =  XGBRegressor() 

#**Train the model with training sets

lr.fit(X_train,y_train)
svm.fit(X_train,y_train)
rf.fit(X_train,y_train)
gbr.fit(X_train,y_train)
xg.fit(X_train,y_train)

#**Prediction on the test/validation data
lr_prediction = lr.predict(X_test)
svm_prediction = svm.predict(X_test)
rf_prediction = rf.predict(X_test)
gbr_prediction = gbr.predict(X_test)
xg_prediction = xg.predict(X_test)

#**Evaluate model performance
#RMSE is a measure of the differences between the predicted values by the model and the actual values
lr_rmse = mean_squared_error(y_test, lr_prediction, squared=False)
svm_rmse = mean_squared_error(y_test, svm_prediction, squared=False)
rf_rmse = mean_squared_error(y_test, rf_prediction, squared=False)
gbr_rmse = mean_squared_error(y_test, gbr_prediction, squared=False)
xg_rmse = mean_squared_error(y_test, xg_prediction, squared=False)

#**Display the evaluation results
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Support Vector Machine RMSE: {svm_rmse}")
print(f"Random Forest RMSE: {rf_rmse}")
print(f"Gradient Boosting Regressor RMSE: {gbr_rmse}")
print(f"XGBRegressor RMSE: {xg_rmse}")

#**Choose the best model (min rmsevalue)
models = [lr, svm, rf, gbr, xg]
rmse_values = [lr_rmse, svm_rmse, rf_rmse, gbr_rmse, xg_rmse]
best_model_index = rmse_values.index(min(rmse_values))
best_model_object = models[best_model_index]

#**Plot the bar chart to visualize the best model among all
models = ['Linear Regression', 'Support Vector Machine', 'Random Forest', 'Gradient Boosting Regressor', 'XGBRegressor']
plt.figure(figsize=(12,8))
bars = plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'purple', 'orange'])

# Add RMSE values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model RMSE Comparison')
plt.xticks(rotation=45)  # Rotate model names for better visibility
plt.tight_layout()
# Display the chart
plt.show()

#**Retrain the model on entire dataset
linearRegression_Final = LinearRegression()
linearRegression_Final.fit(X_Scaled, Y_scaled)

#D4

#**Save the model
dump(best_model_object, "train_model.joblib")

#**Load the model
loaded_model = load("train.joblib")

#**Create a new test set/ Take input from the user
gender = int(input("Enter gender (0 for female, 1 for male): "))
age = int(input("Enter age: "))
occupation = float(input("Enter occupation: "))
city_category = float(input("Enter City Occupation: "))
marital_status = float(input("Enter marital status: "))

X_test1= scaler.transform([[gender, age, occupation, city_category, marital_status]])

#**Predict the outcome

pred_value= loaded_model.predict(X_test1)
print(pred_value)
print("Predicted Car_Purchase_Amount based on input:",scaler1.inverse_transform(pred_value))

