#Importing Libraries 
import pandas as pd 
import numpy as np 
import matplotlib as mp


#Import Dataset 
data = pd.read_csv('H:\Data\Simple_Linear_Regression\Salary_Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, 1:2]


#Split into train and test 
from sklearn.cross_validation import train_test_split 
xtrain, xtrest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state = 0)



#Fit Linear Regression Model 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)
regressor.score(xtrain,ytrain)

#Predict Salary 
ypred = regressor.predict(9.6)