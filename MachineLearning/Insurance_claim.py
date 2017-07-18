#This is the solution for the Super Data Science workshop on https://www.superdatascience.com/workshop-021-machine-learning-r-insurance-claims/
#Have used Random Forest Classifier to predict wether the customer will claim or not.

#Importing Libraries 
import pandas as pd 
import numpy as np 
np.set_printoptions(threshold=np.nan)
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Import Dataset 
df = pd.read_csv('H:\Data\Data.csv')
x  = df.iloc[:, :-1].values
y  = df.iloc[:, 5:6].values

#Count the number of claims and no Claims 
count_of_claims = df.groupby('claim')['claim'].count()

#Encoding categorical variables like age, bmi and gender
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:,1] = labelencoder.fit_transform(x[:,1])
x[:,2] = labelencoder.fit_transform(x[:,2])
x[:,3] = labelencoder.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features =[1,2,3])
x=onehotencoder.fit_transform(x).toarray()
x=np.delete(x, np.s_[9], axis=1)

#Split data into train and test 
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state = 0)
 
#Create Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 0)
rf.fit(xtrain, ytrain)
ypred = rf.predict(xtest)
ytrainpred = rf.predict(xtrain)


#Confusion Matrix for Train Data Set 
cm_train = confusion_matrix(ytrainpred, ytrain)

#Confusion Matrix for Test Data Set 
cm = confusion_matrix(ypred, ytest)

#Accuracy Score of Test Data Set 
testscore = accuracy_score(ytest, ypred)

#Accuracy Score of Training Data Set 
trainscore = accuracy_score(ytrain, ytrainpred)
