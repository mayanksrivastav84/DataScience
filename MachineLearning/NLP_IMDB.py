#Source: https://www.kaggle.com/c/word2vec-nlp-tutorial/data

import pandas as pd 
import numpy as np
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer 


nltk.download('stopwords')

#Import Data Set 
data = pd.read_csv('labeledTrainData.tsv', delimiter = '\t', header=0, quoting = 3)
test = pd.read_csv('testData.tsv', delimiter = '\t', header=0, quoting = 3)


#Clean Text
nltk.download('stopwords')
corpus = [] 
for i in range (0, 25000):
    review = re.sub(('[^A-Za-z]'), ' ', data['review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000)


train_data_features = vectorizer.fit_transform(corpus)

train_data_features = train_data_features.toarray()

train_data_features.shape

vocab = vectorizer.get_feature_names()
print (vocab)

dist = np.sum(train_data_features, axis=0)

voacaball =[]
for tag, count in zip(vocab, dist):
    vocabb = (count,tag)
    voacaball.append(vocabb)


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit( train_data_features, data["sentiment"] )
result = forest.predict(train_data_features)

output = pd.DataFrame( data={"id":data["id"], "sentiment":result} )

#Test Data Set 
testcorpus = [] 
for i in range (0, 25000):
    testreview = re.sub(('[^A-Za-z]'), ' ', test['review'][i])
    testreview = testreview.lower()
    testreview = testreview.split()
    ps = PorterStemmer()
    testreview = [ps.stem(word) for word in testreview if not word in stopwords.words('english')]
    testreview = ' '.join(testreview)
    corpus.append(testreview)

testvectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000)


test_data_features = testvectorizer.fit_transform(corpus)

test_data_features = test_data_features.toarray()

test_data_features.shape

testvocab = testvectorizer.get_feature_names()

dist = np.sum(test_data_features, axis=0)

from sklearn.ensemble import RandomForestClassifier
testforest = RandomForestClassifier(n_estimators = 100) 
testforest = testforest.fit( test_data_features, test["sentiment"] )
testresult = testforest.predict(test_data_features)

output = pd.DataFrame( data={"id":data["id"], "sentiment":result} )
