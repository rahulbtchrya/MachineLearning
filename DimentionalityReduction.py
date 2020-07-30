# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:29:24 2020

@author: rahul


Here we are implementing some dimentionality reduction for the auto-insurance prediction problem

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, f1_score

from sklearn.model_selection import train_test_split, KFold, cross_val_score


df = pd.read_csv("Auto_Data.csv")
#print(df.head())
org_df = df.copy()


# define function for getting corelation variables
def find_corr_features(df_corr, pos_corr_thresh = 0.5, neg_corr_thresh = -0.5):
  corr_feat_pos = []
  corr_feat_neg = []
  for idx in df_corr.index:
    for col in df_corr.columns:
      if idx == col:
        break
      # print("corr_values[{},{}]={}".format(idx,col,corr_values.loc[idx,col]))
      if df_corr.loc[idx,col] < neg_corr_thresh:
        # print("df_corr[{},{}]={}".format(idx,col,df_corr.loc[idx,col]))
        # print("Found negative correlation")
        corr_feat_neg.append((idx,col,df_corr.loc[idx,col]))
      elif df_corr.loc[idx,col] > pos_corr_thresh:
        # print("df_corr[{},{}]={}".format(idx,col,df_corr.loc[idx,col]))
        # print("Found positive correlation")
        corr_feat_pos.append((idx,col,df_corr.loc[idx,col]))
  return (corr_feat_pos,corr_feat_neg)


# define function for plotting pie charts for categorical columns
def plot_pie_charts(df_to_plot, cols_to_plot):
  for col in cols_to_plot:
    plt.figure()
    df_to_plot[col].value_counts().plot(kind='pie', figsize=(5,5), autopct='%1.2f%%')
    plt.title(col)



# **************** Clean the data *******************
# drop duplicates
df = df.drop_duplicates()
print(df.shape)

df.drop('id',axis=1, inplace=True)  # drop the id column as its not needed

# find out if there are any null/missing values
# Note - In this example the value -1 is treated as null. So we replace all -1 values with np.nan in the dataset
print(df.isin(['-1']).sum(axis=0))
df.replace(-1,np.nan,inplace=True)
print(df.isnull().sum())
#print(df.isna().sum())
# print percentages
print(df.isnull().sum()/len(df) * 100)

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
print(df.isnull().sum()/len(df) * 100)

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

# plot pycharts for all categorical columns to check for skewed data
plot_pie_charts(df, cat_cols)
# drop all columns which are more than 90% skewed
df.drop(['ps_car_07_cat', 'ps_car_10_cat'], axis=1,inplace=True)
cat_cols.remove('ps_car_07_cat')
cat_cols.remove('ps_car_10_cat')

# plot pycharts for all binary columns to check for skewed data
plot_pie_charts(df, bin_cols)
# drop all columns which are more than 90% skewed
df.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'], axis=1,inplace=True)
bin_cols.remove('ps_ind_10_bin')
bin_cols.remove('ps_ind_11_bin')
bin_cols.remove('ps_ind_12_bin')
bin_cols.remove('ps_ind_13_bin')

# Split the data into train and test sets
train_df, test_df, label_train, label_test = train_test_split(df, label, random_state=101, stratify=label, test_size=0.2)

# Now we have to impute those values which are missing
# For continuous values we would impute them with mean/median
imputer1 = SimpleImputer(missing_values= np.nan, strategy='mean')
train_df[num_cols] = imputer1.fit_transform(train_df[num_cols])
test_df[num_cols] = imputer1.transform(test_df[num_cols])
print(train_df.isnull().sum()/len(train_df) * 100)
# For categorical variables, we would impute them with their mode
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train_df[cat_cols] = imputer2.fit_transform(train_df[cat_cols])
test_df[cat_cols] = imputer2.transform(test_df[cat_cols])
print(train_df.isnull().sum()/len(train_df) * 100)
# Now there are no missing values. 
# Note that for training data we use fit_transform, but for test-data we use only transform

# Use the low variance method to remove columns having low variance
# First normaize the data. Note that Standard scaler cannot be used here since it makes all variances uniform
train_df[num_cols] = normalize(train_df[num_cols])
variance = train_df[num_cols].var()
# drop all variables having variance less than 0.006
drop_vari = []
for i in range(0, len(variance)):
    if variance[i] <= 0.006:
        drop_vari.append(num_cols[i])
     
train_df.drop(drop_vari,axis=1, inplace=True)
test_df.drop(drop_vari,axis=1, inplace=True)
for item in drop_vari:
    num_cols.remove(item)


# Now we check for correlation between the nemeric variables and drop any variables having correlation great than 0.5 or 0.6
# plot correlation matrix for numeric columns
corr_matrix = df[num_cols].corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix,cmap="BrBG",annot=True)

# We remove one of the variables having correlation more than 0.5
# Note - here we dont have any highly corelated features as seen in the correlation matrix
corr_feat_pos_all, corr_feat_neg_all = find_corr_features(corr_matrix)

if len(corr_feat_pos_all) > 0:
  print("Features with positive correlation:")
  print(corr_feat_pos_all)
  for (f1,f2,corr) in corr_feat_pos_all:
    print("Dropping feature={}".format(f1))
    train_df = train_df.drop(columns=f1, errors='ignore')
    test_df = test_df.drop(columns=f1, errors='ignore')
    num_cols.remove(f1)
else:
  print("0 features with positive correlation")

if len(corr_feat_neg_all) > 0:
  print("Features with negative correlation:")
  print(corr_feat_neg_all)
  for (f1,f2,corr) in corr_feat_neg_all:
    print("Dropping feature={}".format(f1))
    train_df = train_df.drop(columns=f1, errors='ignore')
    test_df = test_df.drop(columns=f1, errors='ignore')
    num_cols.remove(f1)
else:
  print("0 features with negative correlation")       
  

#train_df[num_cols].hist()
# Now we apply MinMax scaler to the numerical columns
std_scaler = MinMaxScaler()
train_df[num_cols] = std_scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = std_scaler.transform(test_df[num_cols])
#train_df[num_cols].hist()
                                    
# Now we check if the output label is skewed or not
label_train.value_counts().plot(kind='pie', figsize=(5,5), autopct='%1.2f%%')
# We see that 96% of the labels are skewed towards 0. We need to upsample or down-weight for 1

# Create a logistic regression model.The class_weight='balanced' automatically adjusts the weights for the target classes.
lreg_model = LogisticRegression(max_iter=500,class_weight='balanced')
print("Training log reg model....")
lreg_model.fit(train_df,label_train)
print("Evaluate the model with test results")
score = lreg_model.score(test_df,label_test)
print("Score on test data :", score)
# Make predictions on test data
label_pred = lreg_model.predict(test_df)
print("Classification report:")
print(classification_report(label_test,label_pred))
print("F1 score:", f1_score(label_test,label_pred))


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


#%%
# use Random forest classifier to find out most important features
#https://www.kdnuggets.com/2015/05/7-methods-data-dimensionality-reduction.html
print("Building random forest....")
my_importance_model = RandomForestClassifier(random_state=1, max_depth=10)
my_importance_model.fit(train_df, label_train)
print(my_importance_model.feature_importances_)

print("plotting top 10 fetures......")
plt.figure(figsize=(10,10))
feat_importances = pd.Series(my_importance_model.feature_importances_, index = train_df.columns)
feat_importances.nlargest(10).plot(kind='barh');
plt.xlabel('Relative Importance')

print("Selecting top 10 features .....")
# Select top 10 important features from the model
feature = SelectFromModel(my_importance_model)
Fit = feature.fit_transform(train_df,label_train)

# Now run logistic regression with the new selected list of features
print("Training log reg model....")
lreg_model.fit(Fit,label_train)

test_df = feature.transform(test_df)
print("Evaluate the model with test results")
score = lreg_model.score(test_df,label_test)
print("Score on test data :", score)
# Make predictions on test data
label_pred = lreg_model.predict(test_df)
print("Classification report:")
print(classification_report(label_test,label_pred))
print("F1 score:", f1_score(label_test,label_pred))
