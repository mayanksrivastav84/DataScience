#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:09:47 2017

@author: MayankSrivastava
"""

#Simple Linear Regression

#Importing Libraries 
import numpy as np 
import matplotlib.pyplot as plt #Graph Library
import pandas as pd #Import datasets into python 
np.set_printoptions(threshold=np.nan) #Display all values in array
 
#Importing DataSet 
dataset = pd.read_csv('/Users/MayankSrivastava/Downloads/DataScience/ML/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
x = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values  

#Splitting the dataset into training and test dataset 
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=1/3, random_state = 0)

#Feature Scaling 
"""from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)"""

#Fitting Simple Linear Regression to Training Set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting The test Results 
ypred  = regressor.predict(xtest)

#Visualizing the training Test Results 
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()