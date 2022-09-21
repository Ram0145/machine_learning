# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 08:50:01 2022

@author: Ram
"""
#Main project TFIDF approach
 
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

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=5).fit(features_train) #min_df is a word which is appearing less than 5 docs, ignore that word. it will reduce word count
len(vect.get_feature_names()) #19841

vect.get_feature_names()[11000:11010] #we can give any numbers below 19841
features_train_vectorized = vect.transform((features_train))

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
accuracy_score(labels_test, pred)  #0.9017427086888914

pred = model.predict_proba(features_test_vectorization) #this line gives 2 probability which were calculated using Logistic Regression. The model will pick probability which is greater in both probabilities then it will give output.  This is how Logistic Regression works. 

#converting model into a pickle file
import pickle
file = open("pickle_model.pkl","wb")




