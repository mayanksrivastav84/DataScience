#Importing Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt #Graph Library


#Import Dataset 
kc = pd.read_csv('/Users/MayankSrivastava/Downloads/Housing/kc_house_data.csv')

#Select only the numerical ones
dfdata = kc.select_dtypes(include=['float64', 'int64'])

#Split into Independent and Dependent Variables
x = dfdata.iloc[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
y = dfdata.iloc[:, 1:2].values


#Taking care of missing Data
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:19])
x[:, 1:19] = imputer.transform(x[:, 1:19])

#Splitting data into train and test 
from sklearn.cross_validation import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)


"""#Fitting Multiple Linear Regression to the training Set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.show()

#Predict for Test Data Set 
ytrainpred = regressor.predict(xtrain)
ytestpred = regressor.predict(xtest)"""

#Using Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(xtrain,ytrain)


#Predict for Test Data Set 
ytrainpred = regressor.predict(xtrain)
ytestpred = regressor.predict(xtest)