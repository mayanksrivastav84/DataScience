#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:30:10 2017

@author: MayankSrivastava
"""

#Importing Libraries 
import numpy as np 
import matplotlib.pyplot as plt #Graph Library
import pandas as pd #Import datasets into python 
np.set_printoptions(threshold=np.nan) #Display all values in array
 
#Importing DataSet 
dataset = pd.read_csv('/Users/MayankSrivastava/Downloads/DataScience/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data.csv')
x = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 3].values  

#Splitting the dataset into training and test dataset 
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state = 0)

#Feature Scaling 
"""from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)"""

