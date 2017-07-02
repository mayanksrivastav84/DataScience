#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:29:59 2017

@author: MayankSrivastava
"""
#Polynomial Linear Regression

#Importing Libraries 
import numpy as np 
import matplotlib.pyplot as plt #Graph Library
import pandas as pd #Import datasets into python 
np.set_printoptions(threshold=np.nan) #Display all values in array
 
#Importing DataSet 
dataset = pd.read_csv('/Users/MayankSrivastava/Downloads/DataScience/ML/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values  
y = dataset.iloc[:, 2].values  


#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression 
linreg = LinearRegression()
linreg.fit(x,y)

#fitting polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree=4)
xpoly = polyreg.fit_transform(x)
linreg2 = LinearRegression()
linreg2.fit(xpoly, y)

#Visualize the Linear Regression Results 
plt.scatter(x, y, color= 'red')
plt.plot(x, linreg.predict(x), color = 'blue')
plt.title('Truth vs Bluff(Linear Regression)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#Visualize the Polynomial Regression Results 
plt.scatter(x, y, color= 'red')
plt.plot(x, linreg2.predict(polyreg.fit_transform(x)), color = 'blue')
plt.title('Truth vs Bluff (Polynomial Regression)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#Visualize the Polynomial Regression Results better
xgrid = np.arange(min(x), max(x), 0.1)
xgrid = xgrid.reshape(len(xgrid), 1)
plt.scatter(x, y, color= 'red')
plt.plot(xgrid, linreg2.predict(polyreg.fit_transform(xgrid)), color = 'blue')
plt.title('Truth vs Bluff (Polynomial Regression)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()


#Preidictng a new result with Linear Regression 
linreg.predict(6.5) 

#Preidictng a new result with Polynomial Regression 
linreg2.predict(polyreg.fit_transform(6.5))
