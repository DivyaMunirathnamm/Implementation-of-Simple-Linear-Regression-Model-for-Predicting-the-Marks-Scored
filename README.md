# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 


## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DIVYA M
RegisterNumber: 212223040043
*/
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset=pd.read_csv("student_scores.csv")
print(dataset.head())
print(dataset.tail())
dataset.info()
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
a=np.array([[13]])
y_pred1=reg.predict(a)
print(y_pred1)
```

## Output:
![Screenshot 2025-03-16 103845](https://github.com/user-attachments/assets/b81bdf6e-c239-4462-820c-dfbf112c5fca)
![Screenshot 2025-03-16 103909](https://github.com/user-attachments/assets/ae38e444-caef-4f4c-b630-c8d9a49650be)
![Screenshot 2025-03-16 103925](https://github.com/user-attachments/assets/4ade2f06-ce7c-4421-988b-e6c43ea0167f)
![Screenshot 2025-03-16 103940](https://github.com/user-attachments/assets/8cee403e-eb5b-4cc1-97c8-148e85ccb387)
![Screenshot 2025-03-16 103955](https://github.com/user-attachments/assets/cd6121b8-68dc-4984-bfa9-a7a5e417f16d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
