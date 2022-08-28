# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 23:01:57 2022

@author: Ram
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 22:54:45 2022

@author: Ram
"""
# Linear Regression 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_scores.csv")
df.isnull().any(axis=0)
df.isnull().sum()
df.isnull().any()
df.isnull().all()
df.describe()
print(df.shape, df.columns, df.dtypes, sep="\n\n")

df["Hours"]
type(df["Hours"])

df["Hours"].values #  we add .values im order to remove the index of df series and also to convert from Series to ndarray
type(df["Hours"].values)
df_features = df["Hours"].values

df["Scores"].values
type(df["Scores"].values)
df_lables = df["Scores"].values

plt.scatter(df_features, df_lables)
plt.show()


from sklearn.linear_model import LinearRegression

#call the constructor inorder to allcocate memory to the call
#once the object is allocated to our alogo then we call that as model 

regressor = LinearRegression()

#converting 1D to 2D so that fit function will work
df_features = df_features.reshape(25,1)

"""
regressor.fit(df_features, df_lables)

#slope
m = regressor.coef_
#intercept
c = regressor.intercept_
"""

"""
x = 9 
y = m*x + c
print(y)

x = [x]
x = np.array(x)
x= x.reshape(1,1)  # or instead of x= [x], we can do x = [[x]] then no need to do reshape
a = regressor.predict(x)

pred = regressor.predict(df_features)
plt.scatter(df_features, df_lables)
plt.plot(df_features, pred, color = 'red')
plt.show()



from sklearn.model_selection import train_test_split
x = [1,2,3,4,5,6,7,8,9,10]  #even if we give an string or np array, I will work as same. Basically it has to be an iterable 
train_test_split(x)
train, test = train_test_split(x, test_size=0.2)

import numpy as np
x = np.arange(10)
train_test_split(x)


lst1 = [1,2,3,4,5,6,7,8,9,10]
lst2 = [10,20,30,40,50,60,70,80,90,100]
train_test_split(lst1,lst2)  #we can give multiple iterables

x = [1,2,3,4,5,6,7,8,9,10] 
train_test_split(x, test_size=0.2, random_state=0) #random state will fix the output for one value. If u change value of random state value, it gives different output but that output is fixed for that value
train_test_split(x, test_size=0.2, random_state=1000)
"""
from sklearn.model_selection import train_test_split
feature_train, features_test, lables_train, lables_test = train_test_split(df_features, df_lables, test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(feature_train, lables_train)

#testing the model
pred_values = regressor.predict(features_test)

pd.DataFrame(zip(pred_values, lables_test))

#train score 
regressor.score(feature_train, lables_train)

#test_score
regressor.score(features_test, lables_test)










