#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 19:43:34 2017

@author: MayankSrivastava
"""

#Importing Libraries 
import numpy as np 
import matplotlib.pyplot as plt #Graph Library
import pandas as pd #Import datasets into python 
np.set_printoptions(threshold=np.nan) #Display all values in array
 
#Importing DataSet 
dataset = pd.read_csv('/Users/MayankSrivastava/Downloads/DataScience/ML/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values  
y = dataset.iloc[:, 2].values  

#Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y= sc_y.fit_transform(y)


#Fitting SVR to the dataset 
from sklearn.svm import SVR 
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

#Predicting a new result 
ypred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([6.5]))))


#Visualize the Polynomial Regression Results 
xgrid = np.arange(min(x), max(x), 0.1)
xgrid = xgrid.reshape((len(xgrid), 1))
plt.scatter(x, y, color= 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth vs Bluff (Polynomial Regression)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()