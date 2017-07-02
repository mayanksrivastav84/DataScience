#Importing Libraries 
import pandas as pd 
import numpy as np 
import matplotlib as mp


#Import Dataset 
Ktrain = pd.read_csv('/Users/MayankSrivastava/Downloads/Housing/Data/train.csv')
Ktest = pd.read_csv('/Users/MayankSrivastava/Downloads/Housing/Data/test.csv')

#Check data types in the dataset 
Ktrain.dtypes

#Select only the numerical ones
dfdata = Ktrain.select_dtypes(include=['float64', 'int64'])
dfactual = Ktest.select_dtypes(include=['float64', 'int64'])

#Split into Independent and Dependent Variables
x = dfdata.iloc[:, :-1].values
y = dfdata.iloc[:, 37].values
z = dfdata.iloc[:, :37]

#Taking care of missing Data
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:36])
x[:, 1:36] = imputer.transform(x[:, 1:36])

#Splitting data into train and test 
from sklearn.cross_validation import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)


#Fitting Multiple Linear Regression to the training Set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predict for Test Data Set 
ypred = regressor.predict(xtest)


#Build model for backward elimination
import statsmodels.formula.api as sm 
x = np.append(arr = np.ones((1460,1)).astype(int), values = x, axis =1)
x_opt = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()


x_opt = x[:, [0,2,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()


x_opt = x[:, [0,2,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()


x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()


x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,17,18,19,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,17,18,19,20,21,22,23,24,25,26,27,31,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,17,18,19,20,21,22,23,24,25,26,27,32,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,17,18,19,20,21,22,23,24,25,26,27,33,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()


x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,17,18,19,20,21,22,23,24,25,26,27,34,35,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,17,18,19,20,21,22,23,24,25,26,27,34,36,37]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()


x_opt = x[:, [0,2,5,6,7,8,9,10,14,15,17,18,19,20,21,22,23,24,25,26,27,34]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()


#Fitting Multiple Linear Regression to the training Set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(x_opt, ytrain)
