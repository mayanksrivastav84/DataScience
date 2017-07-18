#Importing Libraries 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib as plt 

#Import Dataset 
df = pd.read_csv('train.csv')

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

#Visualization of the correlation of variable with SalePrice
dfnum.plot.scatter(x = '1stFlrSF', y = 'SalePrice')

dfnum.plot.scatter(x = 'GrLivArea', y = 'SalePrice')

sns.boxplot(x = 'GarageCars', y = 'SalePrice', data = dfnum)

dfnum.plot.scatter(x = 'GarageArea', y = 'SalePrice')

dfnum.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice')

dfnum.plot.scatter(x = '1stFlrSF', y = 'SalePrice')

sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = dfnum)

sns.boxplot(x = 'YearRemodAdd', y = 'SalePrice', data = dfnum)

sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = dfnum)

sns.boxplot(x = 'FullBath', y = 'SalePrice', data = dfnum)

sns.boxplot(x = 'TotRmsAbvGrd', y = 'SalePrice', data = dfnum)

dfnumcorr = dfnum.corr()
sns.heatmap(dfnumcorr, vmax=.8, square = True)

cols = dfnumcorr.nlargest(5, 'SalePrice')['SalePrice'].index


sns.set()
sns.pairplot(dfnum[cols], size = 2.5)
