# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 21:37:28 2022

@author: Ram
"""

#Feature Selection

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

#Backward elimination
import statsmodels.api as sm
features = sm.add_constant(features)

features_copy = features[:,[0,1,2,3,4,5]]   #we can also write features[:,:]; just to keep a track we are writing like this


regressor_ols = sm.OLS( endog = labels, exog = features_copy).fit()

# now we need to drop unnecessary features
regressor_ols.summary()


#after looking into p-value, dropping the x2
features_copy = features[:,[0,1,3,4,5]]
regressor_ols = sm.OLS( endog = labels, exog = features_copy).fit()
regressor_ols.summary()

#next dropping
features_copy = features[:,[0,1,3,5]]
regressor_ols = sm.OLS( endog = labels, exog = features_copy).fit()
regressor_ols.summary()

#next dropping
features_copy = features[:,[0,3,5]]
regressor_ols = sm.OLS( endog = labels, exog = features_copy).fit()
regressor_ols.summary()

#next dropping
features_copy = features[:,[0,5]]
regressor_ols = sm.OLS( endog = labels, exog = features_copy).fit()
regressor_ols.summary()

#now all the features are having p-value less than 5%

#To get the whole p-value
regressor_ols.pvalues
