#Importing Libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib as plt 
import sklearn

#Import Dataset 
df = pd.read_csv('train.csv')
df1 = pd.read_csv('test.csv')

#Understand the data
df.describe()

#Understand the distribution of Sale Price Data 
print(df['SalePrice'].describe())

#Plot the SalePrice to understand the skewness of data
sns.distplot(df[['SalePrice']], color = 'g', bins = 100)

#Select numerical data types 
set(df.dtypes.tolist())
dfnum = df.select_dtypes(include = ['float64', 'int64'])
dfnum.info()

#Plot all the numerical variables 
dfnum.hist(figsize = (16,20), bins = 50, xlabelsize =8, ylabelsize = 8)

#Find correlation 
dfnum_corr = dfnum.corr()['SalePrice'][:-1]
golden_feature_list = dfnum_corr[abs(dfnum_corr) > 0.5].sort_values(ascending = False)
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_feature_list), golden_feature_list))


#Select attributes in new data frame with higher Correlaton Co-effecient 
data = df.filter(['OverallQual',	'GrLivArea',	'GarageCars',	'GarageArea'	,'TotalBsmtSF'	,'1stFlrSF',	 'FullBath',	'TotRmsAbvGr',	'YearBuilt	',
'YearRemodAd', 'SalePrice'], axis = 1)

test = df1.filter(['OverallQual',	'GrLivArea',	'GarageCars',	'GarageArea'	,'TotalBsmtSF'	,'1stFlrSF',	 'FullBath',	'TotRmsAbvGr',	'YearBuilt	',
'YearRemodAd', 'SalePrice'], axis = 1)

xtrain = dfnum.iloc[:, :-1].values
ytrain = data.iloc[:, 7:8].values
xtest = test.iloc[:, 0:7].values

dfnum.info()

#Handle missing values in Test Data Set 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(xtest[:, 1:7])
xtest[:, 1:7] = imputer.transform(xtest[:, 1:7])

#Feature Scaling for Random Forest
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

#Fit Random Forest Regression 
from sklearn.ensemble import RandomForestRegressor 
rfregressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
rfregressor.fit(xtrain, ytrain)
rfregressor.score(xtrain, ytrain)

#Predict Value of the house
rfytrainpred = rfregressor.predict(xtrain)

#Predict Value of the house on test data set 
rfytestpred = rfregressor.predict(xtest)
