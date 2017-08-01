# Any results you write to the current directory are saved as output.
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib as plt 
import sklearn

#Import Dataset 
df = pd.read_csv('train.csv')
df1 = pd.read_csv('test.csv')


#Select numerical data types 
set(df.dtypes.tolist())
dfnum = df.select_dtypes(include = ['float64', 'int64'])
dfnum.info()

#Delete the outliers
dfnum = dfnum.drop(dfnum[dfnum['Id'] == 1299].index)
dfnum = dfnum.drop(dfnum[dfnum['Id'] == 524].index)


#Select attributes in new data frame with higher Correlaton Co-effecient 
data = df.filter(['OverallQual',	'MSSubClass', 'GrLivArea',	'GarageArea'	,'TotalBsmtSF',	'YearBuilt','YearRemodAdd','SalePrice'], axis = 1)

test = df1.filter(['OverallQual',	'MSSubClass', 'GrLivArea',	'GarageArea'	,'TotalBsmtSF',	'YearBuilt', 'YearRemodAdd'], axis = 1)

xtrain = data.iloc[:, :-1].values
ytrain = data.iloc[:, 7].values
xtest = test.iloc[:, 0:7].values

#Handle missing values in Training Data Set 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(xtrain[:, 0:6])
xtrain[:, 0:6] = imputer.transform(xtrain[:, 0:6])


#Handle missing values in Test Data Set 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(xtest[:, 1:7])
xtest[:, 1:7] = imputer.transform(xtest[:, 1:7])


##Encoding categorical Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
xtrain[:,0] = labelencoder_x.fit_transform(xtrain[:,0])
onehotencoder = OneHotEncoder(categorical_features =[0])
xtrain=onehotencoder.fit_transform(xtrain).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
xtrain[:,1] = labelencoder_x.fit_transform(xtrain[:,1])
onehotencoder = OneHotEncoder(categorical_features =[1])
xtrain=onehotencoder.fit_transform(xtrain).toarray()

##Encoding categorical Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
xtest[:,0] = labelencoder_x.fit_transform(xtest[:,0])
onehotencoder = OneHotEncoder(categorical_features =[0])
xtest=onehotencoder.fit_transform(xtest).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
xtest[:,1] = labelencoder_x.fit_transform(xtest[:,1])
onehotencoder = OneHotEncoder(categorical_features =[1])
xtest=onehotencoder.fit_transform(xtest).toarray()

import xgboost as xgb

regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.7,
                 max_depth=5,
                 min_child_weight=1.5,
                 n_estimators=7429,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

regr.fit(xtrain, ytrain)

# Run predictions
y_pred = regr.predict(xtrain)
y_test = regr.predict(xtest)



##Fit Linear Regression Model 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
regressor.score(xtrain, ytrain)


#Predict Value of the house
ytrainpred = regressor.predict(xtrain)

#Predict Value of the house on test data set 
ytestpred = regressor.predict(xtest)

#Average out the predicted value
finalpred = (y_test+ytestpred)/2

#Output to csv
pred_df = pd.DataFrame(ytestpred, index=df1["Id"], columns=["SalePrice"])
pred_df.to_csv('output.csv', header=True, index_label='Id')


