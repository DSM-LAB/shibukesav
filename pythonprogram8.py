import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
data_set= pd.read_csv('DMVWrittenTests.csv')  
x= data_set.iloc[:, [0,1]].values  
y= data_set.iloc[:, 2].values  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.10)  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)
from sklearn.naive_bayes import GaussianNB  
classifier = GaussianNB()  
classifier.fit(x_train, y_train)  
y_pred= classifier.predict(x_test)
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
print("Accuracy : ", accuracy_score(y_test,y_pred))
print(cm)
