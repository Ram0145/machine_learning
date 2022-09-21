# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 13:03:10 2022

@author: Ram
"""

#missing values 
import numpy as np
import pandas as pd

df = pd.read_csv("cricket_salary_data.csv")
df.shape
df.isnull().any().sum()
df.isnull().sum()
feature = df.values

from sklearn.impute import SimpleImputer #impute means handling missing
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

feature[:,1:2] = imputer.fit_transform(feature[:,1:2])
