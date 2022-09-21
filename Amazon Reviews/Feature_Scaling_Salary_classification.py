# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 08:47:05 2022

@author: Ram
"""

#Feature Scaling

import numpy as np
import pandas as pd

df = pd.read_csv('Salary_Classification.csv')

df.shape
df.isnull().any()
df.columns.tolist()
df.dtypes
df['Department'].unique()
df['Department'].value_counts()

features = df.iloc[:,0:4].values  #index column will be dropped if we include .values
labels = df.iloc[:,4].values

#converting categorical to numerical

#We use One Hot encoding 

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cTrans = ColumnTransformer([('encoder',OneHotEncoder(), [0])], remainder='passthrough')
features = np.array(cTrans.fit_transform(features), dtype = np.float32)

#instead of Column transform, we can use get_dummies from pandas as well. 

features = features[:,1:]

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

#multiple linear regression 
from sklearn.linear_model import LinearRegression
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#as we already splitted into train and test

features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

regressor = LinearRegression()

regressor.fit(features_train, labels_train)

#testing the model 

pred = regressor.predict(features_test)

df_comp = pd.DataFrame(zip(labels_test, pred))

#predicting with new feature
a = ['Development', 1100, 2, 3]
#if we give "Operations" (some thing which is not in dataset) in the above list, 
#it will through an error at cTrans.transform because fit method which was applied on features dont know about this operations and it wont accept it.

#Changing to nd array
a = np.array(a)
#a is with 4 rows and 1 column, that has to be changed to 1 row 4 columns
a = a.reshape(1,4)
#categorical data has to be changed into numeric, so calling already created cTrans
a_transformed = np.array(cTrans.transform(a), dtype = np.float32)
a_transformed = a_transformed[:,1:]
a_transformed = sc.transform(a_transformed)
regressor.predict(a_transformed)









