import numpy as nm          
import matplotlib.pyplot as mtp
import pandas as pd
data_set = pd.read_csv('DMVWrittenTests.csv')
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])  
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))
