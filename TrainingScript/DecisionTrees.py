#Decision Tree Regression

#Importing Libraries 
import numpy as np 
import matplotlib.pyplot as plt #Graph Library
import pandas as pd #Import datasets into python 
np.set_printoptions(threshold=np.nan) #Display all values in array
 
#Importing DataSet 
dataset = pd.read_csv('/Users/MayankSrivastava/Downloads/DataScience/ML/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values  
y = dataset.iloc[:, 2].values  

"""#Splitting the dataset into training and test dataset 
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=1/3, random_state = 0)"""

"""#Feature Scaling 
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
x = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)"""

#Fitting Simple Linear Regression to Training Set 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)


#Predicting The test Results 
ypred  = regressor.predict(6.5)

#Visualizing the Results
xgrid = np.arange(min(x), max(x), 0.01)
xgrid = xgrid.reshape((len(xgrid),1))
plt.scatter(x,y, color = 'red')
plt.plot(xgrid, regressor.predict(xgrid), color = 'blue')
plt.show()

