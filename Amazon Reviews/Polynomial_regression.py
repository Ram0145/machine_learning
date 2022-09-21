# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 18:29:37 2022

@author: Ram
"""

import numpy as np
import pandas as pd

df = pd.read_csv("Claims_Paid.csv")
df.shape
df.isnull().any().sum()
df.isnull().sum()

feature = df.iloc[:,0:1].values
labels = df.iloc[:,1:2].values #instead of this, we can write df.iloc[:,-1]   but this will give lables as 1 dim but the written code ill give the labelas as 2 dim


import matplotlib.pyplot as plt
plt.scatter(feature, labels)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#As there areonly 10 data points, we are not performing train_test_split.  Total data is given as training data 
regressor.fit(feature,labels)

#predicting
x = [1981]
x = np.array(x)
x =x.reshape(1,1)
regressor.predict(x) #106.03933333    
#our model predict 1981 cost is less than 1980.  So this model is not giving accurate enough prediction. 

plt.scatter(feature, labels)
plt.plot(feature, regressor.predict(feature))

regressor.score(feature,labels)

#polynomial regression
#converting to higher degree
regressor_poly = LinearRegression()
#converting to higher degree
from sklearn.preprocessing import PolynomialFeatures
higher_degree_gen = PolynomialFeatures(degree = 5)
higher_degree_features = higher_degree_gen.fit_transform(feature)

regressor_poly.fit(higher_degree_features,labels)

#predicting
x = [1981]
x = np.array(x)
x =x.reshape(1,1)
x = higher_degree_gen.transform(x)
regressor_poly.predict(x)


plt.scatter(feature, labels)
plt.plot(feature, regressor_poly.predict(higher_degree_features)) 

regressor_poly.score(higher_degree_features,labels)
