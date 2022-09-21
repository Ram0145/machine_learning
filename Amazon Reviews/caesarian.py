# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 19:09:15 2022

@author: Ram
"""

#classification

import numpy as np
import pandas as pd

df = pd.read_csv('caesarian.csv')

df.shape
df.isnull().any()
df.columns.tolist()
df.dtypes

features = df.iloc[:,0:5].values
label = df.iloc[:,5].values

#train test split
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, label, test_size=0.2)

#model building - classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier( n_neighbors= 5 , p = 2) #p =2 is Euclidian distance
classifier.fit(features_train, labels_train)

#predicting 
x = [25,1,0,2,1]
x = np.array(x)
x = x.reshape(1,5)
classifier.predict(x)

pred = classifier.predict(features_test)
df_comp = pd.DataFrame(zip(labels_test, pred))
df_comp[0].equals(df_comp[1])

from sklearn.metrics import confusion_matrix
confusion_matrix(labels_test, pred)
