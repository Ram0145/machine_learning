# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 10:14:03 2022

@author: Ram
"""

#Main project 

import pandas as pd
import numpy as np

df = pd.read_csv("balanced_reviews.csv")
df.shape

df.columns.tolist()  #['overall', 'reviewText', 'summary']
df.head()
df['reviewText'].head()
df['reviewText'][0]
df['overall'].unique()
df['overall'].value_counts()
df.isnull().any(axis=0)
df.isnull().sum()

df.dropna(inplace=True)
df['overall'].value_counts()
df.shape

#as we are predicting either the review is positive or negative, we are dropping reviews with rating 3

df['overall'] ==3
(df['overall'] ==3).sum()
df = df[df['overall']!=3]
df['overall'].unique()

df['Positivity'] = np.where(df['overall']>3, 1,0)
df['Positivity'].sample(10)

df['Positivity'].value_counts()

feature = df['reviewText']
labels =df['Positivity']

from sklearn.model_selection import train_test_split
features_train , features_test, labels_train, labels_test = train_test_split(feature, labels, test_size=0.2)

#All algo's work on numeric data but here we are having text data. 
#we need to convert features_train data to numeric representation and this process is called as 'Vectorization'

#We are using Count Vectorization
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer().fit(features_train)   #vocalbulary happens here 

features_train_vectorized = vect.transform((features_train)) #multi hot encoding or sparse matrix creation happens here.

#to see elements inside sparse matrix
#a = features_train_vectorized.toarray()
     #due to lessmemory unable to execute this
     
#to see all unique words
vect.get_feature_names()
len(vect.get_feature_names()) #68194

vect.get_feature_names()[11000:11010]  #we can give any numbers below 68266

#we are using Logistic Regression (works best for NL data)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)

#predicting  0-negative 1-positive
features_test_vectorization = vect.transform(features_test)
pred = model.predict(features_test_vectorization)

from sklearn.metrics import confusion_matrix
confusion_matrix(labels_test, pred)

from sklearn.metrics import accuracy_score
accuracy_score(labels_test, pred)  #0.8990404672507301

pred = model.predict_proba(features_test_vectorization) #this line gives 2 probability which were calculated using Logistic Regression. The model will pick probability which is greater in both probabilities then it will give output.  This is how Logistic Regression works. 

#using kNN

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
model_knn.fit(features_train_vectorized, labels_train)

#These below code crashed the laptop.  
#pred_knn = model_knn.predict(features_test_vectorization)

#from sklearn.metrics import confusion_matrix
#confusion_matrix(labels_test, pred_knn)

#from sklearn.metrics import accuracy_score
#accuracy_score(labels_test, pred_knn)












