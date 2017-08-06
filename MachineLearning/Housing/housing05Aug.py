#Import the required Libraries 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib as plt

# Set ipython's max row display
pd.set_option('display.max_row', 10000)

#Setting to print all the values in array
np.set_printoptions(threshold=np.nan)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 500)

#Import the data into Dataframe 
traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

#Delete the outliers
traindata = traindata.drop(traindata[traindata['Id'] == 1299].index)
traindata = traindata.drop(traindata[traindata['Id'] == 524].index)

#Let's drop the columns which make no sense to keep due to large null values
traindata = traindata.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
testdata = testdata.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)


#On basis of EDA we did earlier, filter out the variable we want to use for predicting the sale price

finaldata = traindata.filter([ 
       'OverallQual',	'MSSubClass', 'KitchenAbvGr','OverallCond', 'GrLivArea', 'EnclosedPorch', 'GarageArea', 'OverallCond'	,'TotalBsmtSF',  'YearBuilt', 'SalePrice'], axis = 1)


finaltest = testdata.filter([ 
       'OverallQual',	'MSSubClass', 'KitchenAbvGr', 'OverallCond','GrLivArea', 'EnclosedPorch', 'GarageArea', 'OverallCond'	,'TotalBsmtSF',  'YearBuilt'], axis = 1)



#Handle mising values in test data 
finaltest.loc[finaltest.GarageArea.isnull(), 'GarageArea'] = 0
finaltest.loc[finaltest.TotalBsmtSF.isnull(), 'TotalBsmtSF'] = 0


#Transform Sale Price and GrLivArea to reduce standardize the data 
finaldata['SalePrice'] = np.log(finaldata['SalePrice'])
finaldata['GrLivArea'] = np.log(finaldata['GrLivArea'])
finaldata['TotalBsmtSF'] = np.log1p(finaldata['TotalBsmtSF'])
finaltest['GrLivArea'] = np.log(finaltest['GrLivArea'])
finaltest['TotalBsmtSF'] = np.log1p(finaltest['TotalBsmtSF'])


#Find out the columns which are missing in final data 

total = finaldata.isnull().sum().sort_values(ascending=False)
percent = (finaldata.isnull().sum()/finaldata.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


#Splt into predictor and variable
xtrain = finaldata.iloc[:, :-1].values
ytrain = finaldata.iloc[:,10].values
xtest = finaltest.iloc[:, :10].values


#Prediction Model
import xgboost as xgb
regr = xgb.XGBRegressor()
regr.fit(xtrain, ytrain)

regr.score(xtrain,ytrain)

# Run predictions using XGBoost
y_pred = regr.predict(xtrain)


#Predict the prices for Test Data Set
y_test = regr.predict(xtest)

##Fit Linear Regression Model 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
regressor.score(xtrain,ytrain)

#Predict Value of the house using Linear Regression
ytrainpred = regressor.predict(xtrain)

#Predict Value of the house on test data set 
ytestpred = regressor.predict(xtest)

#Average out the predicted value
finalpred = (y_test+ytestpred)/2
finalpred = np.exp(finalpred)

#Output to csv
pred_df = pd.DataFrame(finalpred, index=testdata["Id"], columns=["SalePrice"])
pred_df.to_csv('output.csv', header=True, index_label='Id')
