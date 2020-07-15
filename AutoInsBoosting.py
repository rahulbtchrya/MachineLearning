# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:02:56 2020

@author: rahul

Here we will be using the boosting algorithms on the AutoML dataset

"""

import pandas as pd

from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBClassifier
import xgboost as xgb


df = pd.read_csv("AutoML_Data.csv")
#print(df.head())
org_df = df.copy()

# **************** Clean the data *******************
# drop duplicates
df = df.drop_duplicates()
print(df.shape)

df.drop('id',axis=1, inplace=True)  # drop the id column as its not needed

# find out if there are any null/missing values.In this example the value -1 is treated as null.
print(df.isin(['-1']).sum(axis=0))

label = df.pop('target')


# ************* Exploratory data analysis and pre-processing ****************************
# NOTE - Since XGBoost woks with only numerical columns, we dont need to change the cat and bin columns to categorical
# NOTE - Since XGBoost automatically handles missing values, we dont need to do any imputation

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

# Since the data is highly skewed ie. 97% for 0, we need to scale for the postive weights ie. 1 
SCALE_FACTOR = label.value_counts()[0] / label.value_counts()[1]

# Note - In this example the value -1 is treated as null. But XGB can handle these values using the 'missing' parameter.
#        Note that this parameter takes only float values, no strings. Default value is np.nan

# Create XGBoost classifier
clf = XGBClassifier(scale_pos_weight=SCALE_FACTOR, learning_rate=0.01, objective='binary:logistic', missing=-1)

print("Fitting XGBoost model....")
clf.fit(train_df, label_train)

test_score = clf.score(test_df, label_test)
predict_test = clf.predict(test_df)
test_f1_score = f1_score(label_test, predict_test)
roc_auc_score_test = roc_auc_score(label_test,predict_test)
print(f"Test score={test_score} and test f1 score={test_f1_score} and roc_auc_score={roc_auc_score_test}")
print("Test Classification report:")
print(classification_report(label_test,predict_test))

train_score = clf.score(train_df, label_train)
predict_train = clf.predict(train_df)
train_f1_score = f1_score(label_train, predict_train)
roc_auc_score_train = roc_auc_score(label_train,predict_train)
print(f"Train score={train_score} and train f1 score={train_f1_score} and roc_auc_score={roc_auc_score_train}")
print("Train Classification report:")
print(classification_report(label_train, predict_train))

# Now we will try to tune the hyper-parameters of the classifier to improve accuracy
# https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7

# We perform grid search validation to determinen the optimum hyper-parameters
# Lets start with max-depth. We could try with multiple parameters together, but it will take hours
parameters = {
     # "learning_rate"    : [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
       "max_depth"        : [2,6,9]
     # "min_child_weight" : [ 1, 3, 5, 7 ],
     # "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     # "subsample"        : [i/10.0 for i in range(6,10)],
     # "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
     # "reg_alpha"        : [0, 0.001, 0.005, 0.01, 0.05]
     }

grid = GridSearchCV(clf,
                    parameters, n_jobs= -1,
                    scoring='roc_auc',
                    cv=5)   # 5-fold stratified cross-validation

grid.fit(train_df, label_train)

print("Best barams:")
print(grid.best_params_)
print("Best score:")
print(grid.best_score_)


# Since we found that the best param for max depth is 6, we can do further refining for that parameter to find a more optimum value
parameters = {     
       "max_depth" : [4,5,6,7,8]
     }

grid = GridSearchCV(clf,
                    parameters, n_jobs= -1,
                    scoring='roc_auc',
                    cv=5)   # 5-fold stratified cross-validation

grid.fit(train_df, label_train)

print("Best barams:", grid.best_params_)
print("Best score:", grid.best_score_)

# From here we find that best max_depth is 5. So we train the model using this parameter
# Create XGBoost classifier
clf = XGBClassifier(scale_pos_weight=SCALE_FACTOR, learning_rate=0.01, 
                    max_depth=5,
                    objective='binary:logistic', missing=-1)


print("Fitting XGBoost model....")
clf.fit(train_df, label_train)

test_score = clf.score(test_df, label_test)
predict_test = clf.predict(test_df)
test_f1_score = f1_score(label_test, predict_test)
roc_auc_score_test = roc_auc_score(label_test,predict_test)
print(f"Test score={test_score} and test f1 score={test_f1_score} and roc_auc_score={roc_auc_score_test}")
print("Test Classification report:")
print(classification_report(label_test,predict_test))

train_score = clf.score(train_df, label_train)
predict_train = clf.predict(train_df)
train_f1_score = f1_score(label_train, predict_train)
roc_auc_score_train = roc_auc_score(label_train,predict_train)
print(f"Train score={train_score} and train f1 score={train_f1_score} and roc_auc_score={roc_auc_score_train}")
print("Train Classification report:")
print(classification_report(label_train, predict_train))

# In a similar fashion, we can tune all the hyper-paremeters as per the grid below
# parameters = {
#      # "learning_rate"    : [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
#        "max_depth"        : [2,6,9]
#      # "min_child_weight" : [ 1, 3, 5, 7 ],
#      # "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#      # "subsample"        : [i/10.0 for i in range(6,10)],
#      # "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
#      # "reg_alpha"        : [0, 0.001, 0.005, 0.01, 0.05]
#      }
