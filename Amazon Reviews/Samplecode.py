# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:27:29 2022

@author: Ram
"""

import pandas as pd
df_reader = pd.read_json("Clothing_Shoes_and_Jewelry.json", chunksize=1000000, lines=True)
counter = 1
for chunk in df_reader:
    new_df = pd.DataFrame(chunk[["overall","reviewText","summary"]])
    new_df1 = new_df[new_df["overall"]==1].sample(4000)
    new_df2 = new_df[new_df["overall"]==2].sample(4000)
    new_df3 = new_df[new_df["overall"]==4].sample(4000)
    new_df4 = new_df[new_df["overall"]==5].sample(4000)
    new_df5 = new_df[new_df["overall"]==3].sample(8000)
    
    #mergeing all the dfs
    new_df6 = pd.concat([new_df1, new_df2, new_df3, new_df4, new_df5], axis = 0, ignore_index=True)
    #converting into csv and saving them in current directory
    new_df6.to_csv(str(counter)+".csv",index=False)
    print(counter)
    counter=counter+1

#reading the generated csv files
from glob import glob
filenames = glob("*.csv")
type(filenames)

dataframes =[]
#appnding all generated csv files into a list
for f in filenames:
    dataframes.append(pd.read_csv(f))
    
# creating a df  and generated csv using the above list 
final_df = pd.concat(dataframes, axis=0, ignore_index= True)
final_df.to_csv("balanced_reviews.csv", index=False)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    