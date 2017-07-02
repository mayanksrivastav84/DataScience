#Decision Tree Regression

#Importing Libraries 
import numpy as np 
import matplotlib.pyplot as plt #Graph Library
import pandas as pd #Import datasets into python 
np.set_printoptions(threshold=np.nan) #Display all values in array

#Importing DataSet 
dataset = pd.read_csv('/Users/MayankSrivastava/Downloads/DataScience/ML/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values  
y = dataset.iloc[:, 4].values  

#Splitting the dataset into training and test dataset 
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.25, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

#Predicting the Test Set Results 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

#Predict the test results 
ypred= classifier.predict(xtest)


#Making the confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)


#Visualizing the test results 
from matplotlib.colors import ListedColormap 
xset, yset = xtrain, ytrain 
