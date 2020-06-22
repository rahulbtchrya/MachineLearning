# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:44:24 2020

@author: rahul
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline



# Define a class to perform clean-up of data
class CleanUpDataset(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self,documents,y=None):
        return self

    def transform(self, x_dataset):
        print("Inside transform....")
        x_dataset = x_dataset.drop_duplicates()     
        print("duplicates dropped")     
        # find out if there are any null/missing values
        # Note - In this example the value -1 is treated as null. So we replace all -1 values with np.nan in the dataset
       # print(x_dataset.isin(['-1']).sum(axis=0))
        x_dataset.replace(-1,np.nan,inplace=True)
        print("-1 replaced")
        #print(x_dataset.isnull().sum())
        
        # print("Before filling.....")
        # # print percentages
        # print(x_dataset.isnull().sum()/len(df) * 100)   
        
        # Since the SimpleImputer is not working with the pipe-line, we impute here itself using pandas
        for ncol in num_cols:
            x_dataset[ncol].fillna(x_dataset[ncol].mean(), inplace=True)         
        for bcol in bin_cols:
            x_dataset[bcol].fillna(x_dataset[bcol].value_counts().idxmax(), inplace=True) #impute with the most frequent value  
        for ccol in cat_cols:
            x_dataset[ccol].fillna(x_dataset[bcol].value_counts().idxmax(), inplace=True) #impute with the most frequent value
                    
        # print("After filling.....")
        # print(x_dataset.isnull().sum()/len(df) * 100)   
        
        df[cat_cols] = df[cat_cols].astype('category')
        df[bin_cols] = df[bin_cols].astype('category')
        print("category set")                
        
        print("returning dataset of shape", x_dataset.shape)               
        return x_dataset
    
  
    
df = pd.read_csv("AutoML_Data.csv")
#print(df.head())
org_df = df.copy()
df.drop('id',axis=1, inplace=True)  # drop the id column as its not needed

label = df.pop('target')

# segregate the features into categorical, binary and numerical features
cat_cols =[] 
num_cols =[]
bin_cols = []

for col_name in df.columns:
    if 'bin' in col_name:
        bin_cols.append(col_name)
    elif 'cat' in col_name:
        cat_cols.append(col_name)
    else:
        num_cols.append(col_name)

df.replace(-1,np.nan,inplace=True)  # testing

print(f"Binary colums={len(bin_cols)} , Categorical columns={len(cat_cols)}, Numeric columns={len(num_cols)}")        

# Split the data into train and test sets
train_df, test_df , label_train , label_test = train_test_split(df, 
                                                                label, 
                                                                random_state=101, # random seed to get the same values everytime 
                                                                stratify=label, # train and test samples will have same no. of label examples
                                                                test_size=0.3 # 30% will be test and 70% train
                                                                )

#Check if the train and test labels have the same percentage-distribution of values
print(label_train.value_counts(normalize=True))
print(label_test.value_counts(normalize=True))


# Define a Column Transformer to do pre-procesing steps
# - Impute values for categroical and numeric columns
# - Scale numerical columns
# - Drop required columns - refer to AutoML for list of columns
pre_process = ColumnTransformer(remainder='passthrough',
                                transformers=[    
                                    # imputers dont work, probably for multiple columns together
                                    # ('impute_nums', SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=True, verbose=2), num_cols),
                                    # ('impute_cats', SimpleImputer(missing_values=np.nan, strategy='most_frequent', add_indicator=True, verbose=2), cat_cols),
                                    # ('impute_bins', SimpleImputer(missing_values=np.nan, strategy='most_frequent', add_indicator=True, verbose=2), bin_cols),
                                    ('scale_nums', MinMaxScaler(), num_cols),
                                    ('drop_columns', 'drop', ['ps_car_07_cat','ps_car_10_cat','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin'])                                                                                                
                                    ]                           
                                ) 
# NOTE - there seems to be an issue with SimpleImputer and pipelines                
                                
# Now define the pipe-line
# 1 - Cleanup the data 
# 2 - Pre-process the data
# 3 - Logistic regression 
pipeline_model = make_pipeline(
                                CleanUpDataset(),
                                pre_process,
                                LogisticRegression(max_iter=100,class_weight='balanced')
                                )

print("Training the pipe-line.....")
pipeline_model.fit(train_df, label_train)

print("Evaluate the model with test results....")
score = pipeline_model.score(test_df,label_test)
print("Score on test data :", score)
# Make predictions on test data
label_pred = pipeline_model.predict(test_df)
print("Classification report on test:")
print(classification_report(label_test,label_pred))
print("Test F1 score:", f1_score(label_test,label_pred))

pred_train = pipeline_model.predict(train_df)
print("Train_F1 score:", f1_score(label_train,pred_train))
print("Classification report on train:")
print(classification_report(label_train,pred_train))
print("Score on train data :", pipeline_model.score(train_df,label_train))


    

