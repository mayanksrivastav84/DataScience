#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 22:51:09 2017

@author: MayankSrivastava
"""


#Simple Linear Regression

#Importing Libraries 
import numpy as np 
import matplotlib.pyplot as plt #Graph Library
import pandas as pd #Import datasets into python 
np.set_printoptions(threshold=np.nan) #Display all values in array
 
#Importing DataSet 
dataset = pd.read_csv('/Users/MayankSrivastava/Downloads/DataScience/ML/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
x = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values  

#Splitting the dataset into training and test dataset 
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=1/3, random_state = 0)

