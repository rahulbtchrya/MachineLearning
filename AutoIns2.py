# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:48:37 2020

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


df = pd.read_csv("AutoML_Data.csv")
#print(df.head())
org_df = df.copy()

# **************** Clean the data *******************
# drop duplicates
df = df.drop_duplicates()
print(df.shape)

df.drop('id',axis=1, inplace=True)  # drop the id column as its not needed

# find out if there are any null/missing values
# Note - In this example the value -1 is treated as null. So we replace all -1 values with np.nan in the dataset
print(df.isin(['-1']).sum(axis=0))
df.replace(-1,np.nan,inplace=True)
#print(df.isnull().sum())
#print(df.isna().sum())
# print percentages
#print(df.isnull().sum()/len(df) * 100)

# remove those columns which have more than 60% values missing
# saving missing values in a variable
a = df.isnull().sum()/len(df)*100
# saving column names in a variable
variables = df.columns
drop_variables = [ ]
for i in range(0,len(df.columns)):
    if a[i] > 60:   #setting the threshold as 60%
        drop_variables.append(variables[i])
df.drop(drop_variables, axis=1, inplace=True)
print(df.shape)
#print(df.isnull().sum()/len(df) * 100)

label = df.pop('target')


# ************* Exploratory data analysis and pre-processing ****************************

columns_list = df.columns
# segregate the features into categorical, binary and numerical features
cat_cols = []
num_cols = []
bin_cols = []
for col_name in columns_list:
    if 'bin' in col_name:
        bin_cols.append(col_name)
    elif 'cat' in col_name:
        cat_cols.append(col_name)
    else:
        num_cols.append(col_name)

print(f"Binary colums={len(bin_cols)} , Categorical columns={len(cat_cols)}, Numeric columns={len(num_cols)}")

# Convert categorical and binary columns as categorical data-type
df[cat_cols] = df[cat_cols].astype('category')
df[bin_cols] = df[bin_cols].astype('category')

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

# Now we have to impute those values which are missing
# For continuous values we would impute them with mean/median
# train['Item_Weight'].fillna(train['Item_Weight'].median(), inplace=True)
imputer1 = SimpleImputer(missing_values= np.nan, strategy='mean')
train_df[num_cols] = imputer1.fit_transform(train_df[num_cols])
test_df[num_cols] = imputer1.transform(test_df[num_cols])
#print(train_df.isnull().sum()/len(train_df) * 100)
# For categorical variables, we would impute them with their mode
# train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_df[cat_cols] = imputer2.fit_transform(train_df[cat_cols])
test_df[cat_cols] = imputer2.transform(test_df[cat_cols])
#print(train_df.isnull().sum()/len(train_df) * 100)
# Now there are no missing values. 
# Note that for training data we use fit_transform, but for test-data we use only transform

#train_df[num_cols].hist()
# Now we apply MinMax scaler to the numerical columns
std_scaler = MinMaxScaler()
train_df[num_cols] = std_scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = std_scaler.transform(test_df[num_cols])
#train_df[num_cols].hist()
                                    
# Now we check if the output label is skewed or not
label_train.value_counts().plot(kind='pie', figsize=(5,5), autopct='%1.2f%%')
# We see that 96% of the labels are skewed towards 0. We need to upsample or down-weight for 1

# ***** Create a logistic regression model.The class_weight='balanced' automatically adjusts the weights for the target classes.
lreg_model = LogisticRegression(max_iter=1000,class_weight='balanced')
print("Training logisitc reg model....")
lreg_model.fit(train_df,label_train)
print("Evaluate the model with test results")
score = lreg_model.score(test_df,label_test)
print("Score on test data :", score)
# Make predictions on test data
label_pred = lreg_model.predict(test_df)
print("Classification report:")
print(classification_report(label_test,label_pred))
print("Test F1 score:", f1_score(label_test,label_pred))

pred_train = lreg_model.predict(train_df)
print("Train_F1 score:", f1_score(label_train,pred_train))
print(classification_report(label_train,pred_train))


# ****  Now we check for variables which have high coefficients and consider only those variables
print("Logistic regression coffecients:")
print(lreg_model.coef_)

X = train_df.columns
Xl = range(len(X))
C = lreg_model.coef_.reshape(-1)

# plot the values of the coefficients
plt.figure(figsize=(8,6))
plt.bar(Xl,abs(C))

# Create a data-frame with the variables and their cofficients
coeff_df = pd.DataFrame({
                         'variables': X , 
                         'coefficients': abs(C)
                         })


# Get the list of cofficients which have values greater than a particular value
coeff_cols = coeff_df[(coeff_df.coefficients > 0.3)]

# Take only a subset of train_df which contains only tha variables having high coefficients
train_df = train_df[coeff_cols['variables'].values]
test_df = test_df[coeff_cols['variables'].values]


# Now train the logistic regression model on the reduced subset of columns
print("Training logisitc reg model with selected columns ....")
logisreg_model = LogisticRegression(max_iter=1000,class_weight='balanced')
logisreg_model.fit(train_df,label_train)
print("Evaluate the model with test results")
score = logisreg_model.score(test_df,label_test)
print("Score on test data :", score)
# Make predictions on test data
label_pred = logisreg_model.predict(test_df)
print("Classification report on test data:")
print(classification_report(label_test,label_pred))
print("Test F1 score:", f1_score(label_test,label_pred))

pred_train = logisreg_model.predict(train_df)
print("Train_F1 score:", f1_score(label_train,pred_train))
print("Classification report on train data:")
print(classification_report(label_train,pred_train))


                                           
                       
