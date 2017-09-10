import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode

# Set ipython's max row display
pd.set_option('display.max_row', 10000)
#Setting to print all the values in array
1
np.set_printoptions(threshold=np.nan)
# Set iPython's max column width to 50
pd.set_option('display.max_columns', 500)

#Import Train Dataset for EDA

train = pd.read_csv('/Users/MayankSrivastava/Downloads/MachineLearning/Kaggle/SberBank Housing/train.csv')

train.info()


#Identify the columns with missing Values
total = train.isnull().sum().sort_values(ascending=False)
total.columns = ['column_name', 'missing_count']
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.loc[missing_data['Total']!= 0]


#Understand the distribution of Target Variable Price
train['price_doc'].describe()

np.log(train['price_doc']).skew()

train['price_doc'].kurtosis()

sns.distplot(train['price_doc'], color = 'g', bins = 100)

sns.distplot(np.log(train['price_doc']), color = 'g', bins = 100, kde = 'True')

from scipy import stats

stats.probplot(train['price_doc'], plot = plt)

stats.probplot(np.log1p(train['price_doc']), plot = plt)


#Scatter Plot to understand the relation between price and internal features 

train.plot.scatter(x = 'full_sq', y = 'price_doc')

train.plot.scatter(x = 'life_sq', y = 'price_doc')

train.plot.scatter(x = 'floor', y = 'price_doc')

sns.boxplot(x ='max_floor', y = 'price_doc', data = train)

sns.boxplot(x ='material', y = 'price_doc', data = train)

sns.pointplot(x = 'floor', y = 'price_doc', data = train, alpha = 0.8, color = 'r')

train.plot.scatter(x = 'kitch_sq', y = 'price_doc')

sns.boxplot(x = 'state', y = 'price_doc', data = train)

sns.boxplot(x = 'product_type', y = 'price_doc', data = train)

#Correlation between price and internal features 

internal_features = train[['full_sq', 'life_sq', 'floor', 'build_year', 'num_room', 'material', 'max_floor', 'kitch_sq', 'state', 'price_doc']]

internal_features_corr = internal_features.corr()['price_doc'][:-1]

internal_features_corr_list = internal_features_corr[abs(internal_features_corr > 0)].sort_values(ascending = False)

#Create heatmap for correlation between internal features and Price

internal_features_heatmap = internal_features.corr()

cols = internal_features_heatmap.nlargest(10, 'price_doc')['price_doc'].index

cm = np.corrcoef(internal_features_heatmap[cols].values.T)

sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


#Analyze the internal features of the houses
train.groupby('build_year').build_year.nunique()

#replace value 20052009 in the build year with more realistic value 

train['build_year'] = train.replace(20052009, mode(train['build_year']))

train.groupby('state').state.nunique()
