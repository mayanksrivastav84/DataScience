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

#Taking Care of the missing Data 
from sklearn.preprocessing import Imputer #Library is ScikitLearn class is Imputer. 
imputer = Imputer(missing_values='NaN' , strategy="mean", axis=0)

#Fit imputer object to matrix X
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


#Encoding categorical Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features =[0])
x=onehotencoder.fit_transform(x).toarray()

#Encoding the Purchased. 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#Splitting the dataset into training and test dataset 
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state = 0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

