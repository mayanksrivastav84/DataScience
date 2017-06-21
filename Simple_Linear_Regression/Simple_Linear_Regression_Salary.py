#Importing Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as mp


#Import Dataset 
data = pd.read_csv('H:\Data\Simple_Linear_Regression\Salary_Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, 1:2]


#Split into train and test 
from sklearn.cross_validation import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 1/3, random_state = 0)



#Fit Linear Regression Model 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)
regressor.score(xtrain,ytrain)

#Predict Salary 
ypred = regressor.predict(xtrain)

#Visualizing the Training Set Results 
mp.scatter(xtrain, ytrain, color = 'red')
mp.plot(xtrain, regressor.predict(xtrain), color = 'blue')
mp.title('Salary vs Experience (Training DataSet)')
mp.xlabel('Years of Experience')
mp.ylabel('Salary')
mp.show()

#Visualizing the Test Set Results 
mp.scatter(xtest, ytest, color = 'red')
mp.plot(xtrain, regressor.predict(xtrain), color = 'blue')
mp.title('Salary vs Experience (Training DataSet)')
mp.xlabel('Years of Experience')
mp.ylabel('Salary')
mp.show()
 