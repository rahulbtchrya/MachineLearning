# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:54:02 2020

@author: rahul
"""


"""
Created on Fri Jun 12 13:48:37 2020

@author: rahul
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier


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
#print(df.isin(['-1']).sum(axis=0))
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

# Use Decision Tree classifier
train_accuracy = []
test_accuracy = []
train_f1_score = []
test_f1_score = []
depths = range(1,50)

# Try to check the accuracy for different tree depths
for depth in depths:
    dt_model = DecisionTreeClassifier(random_state=10, max_depth=depth, class_weight='balanced')
    print('Fitting decision tree of depth:' , depth)
    dt_model.fit(train_df,label_train)
    train_accuracy.append(dt_model.score(train_df, label_train))
    test_accuracy.append(dt_model.score(test_df, label_test))    
    train_f1_score.append(f1_score(label_train, dt_model.predict(train_df)))
    test_f1_score.append(f1_score(label_test, dt_model.predict(test_df)))
    

#plot the accuracy for different tree depths
plt.figure()
plt.xlabel("Max_depth")
plt.ylabel("Accuracy")
plt.plot(depths,train_accuracy, label='train_accuracy')
plt.plot(depths, test_accuracy, label='test_accuracy')
plt.legend()
plt.show()

plt.figure()
plt.xlabel("Max_depth")
plt.ylabel("F1 score")
plt.plot(depths,train_f1_score, label='train_f1_Score')
plt.plot(depths, test_f1_score, label='test_f1_Score')
plt.legend()
plt.show()


#%%
# From the graph we see the training and vaidation accuracy are diverging after depth=5
# So we train the decision tree classifier with a depth of 5. If we consider F1 score, we could use max_depth=20

# Use Decision Tree classifier with depth 5
dt_model = None
dt_model = DecisionTreeClassifier(random_state=10, class_weight='balanced', max_depth=5)
                                                                   
print('Fitting decision tree.........')
dt_model.fit(train_df,label_train)
predict_test = dt_model.predict(test_df)
f1score_test = f1_score(label_test, predict_test)
print("Test accuracy:", dt_model.score(test_df, label_test))
print("Test f1 score =", f1score_test)
print("Classification report test:")
print(classification_report(label_test,predict_test))

print("Classification report train:")
print(classification_report(label_train , dt_model.predict(train_df)))
print("Train f1 score =", f1_score(label_train, dt_model.predict(train_df)))
print("Train accuracy:",dt_model.score(train_df, label_train))
#%%
# vizualize the tree
from sklearn import tree
tree.export_graphviz(dt_model, out_file='my_tree.dot', feature_names=train_df.columns, max_depth=5, filled=True)
# convert the tree from .dot file to png file 
# ! dot -Tpng tree.dot -o tree.png - doesn't work
import pydot
(graph,) = pydot.graph_from_dot_file('my_tree.dot')
graph.write_png('my_tree.png')

img = plt.imread('my_tree.png')
plt.figure(figsize=(15,15))
plt.imshow(img)
