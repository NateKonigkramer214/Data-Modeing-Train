import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
print()


#**Transform the output dataset into a percentage based weighted value between 0 and 1
scaler1= MinMaxScaler()
#Reshape
Y_reshape= Y.values.reshape(-1,1)
Y_scaled=scaler1.fit_transform(Y_reshape)


#**Split the dataset into the training set and test set
#! The train_test_split() method is used to split our data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_Scaled, Y_scaled, test_size=0.2, random_state=42)