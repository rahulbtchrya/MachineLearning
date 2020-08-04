# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:03:23 2020

@author: rahul

Here we will use a large spacy model to calculate document vectors and use it along with XGBoost for classification of twitter texts
"""
import pandas as pd
import re
from gensim.parsing import remove_stopwords
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from xgboost.sklearn import XGBClassifier

def clean_data(text):    
    text = re.sub('@[\w]*', '', text)   # remove @user
    text = re.sub('&amp;','',text)             # remove &amp;
    text = re.sub('[?!.;:,,#@-]', '', text)  # remove special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text) # remove Unicode characters
    text = text.replace("[^A-Za-z#]", "") # Replace everything except alphabets and hash
    text = text.lower() # make everything lowercase for uniformity    
    # removing short words which are of length 3 or lower(eg. hmm, oh) since they dont add any value
    text = " ".join(w for w in text.split() if len(w)>3)    
    # removing stop-words eg. 'we', 'our', 'ours', 'ourselves', 'just', 'don', "don't", 'should'
    text = remove_stopwords(text)    
    return text

# *************************  read training data *******************************
df = pd.read_csv('.//data//train_tweets.csv')
print(df.head())

df.drop('id', axis=1, inplace=True)
df.drop_duplicates()
print(df.isna().sum())

tweets = df['tweet']
labels = df['label']

# clean the texts
tweets = tweets.apply(lambda x : clean_data(x))

# Since the data is highly skewed
SCALE_FACTOR = labels.value_counts()[0] / labels.value_counts()[1]

# we load a large model becuase we need vectors
nlp = spacy.load("en_core_web_lg")

# **********************  compute the vectors *********************************

# Disabling other pipes because we don't need them and it'll speed up this part a bit
with nlp.disable_pipes():
    docs = list(nlp.pipe(tweets))
    doc_vectors = np.array([doc.vector for doc in docs])
    
print("doc vectors shape=", doc_vectors.shape)
    
X_train, X_test, y_train, y_test = train_test_split(doc_vectors, labels, test_size=0.3, random_state=1, stratify=labels)

xgb_model=None
xgb_model = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=SCALE_FACTOR,
    objective='binary:logistic',
    colsample=0.9,
    colsample_bytree=0.5,
    eta=0.1,
    max_depth=8,
    min_child_weight=6,
    subsample=0.9)

print("Training xgb model....")
xgb_model.fit(X_train, y_train)

print("score=", xgb_model.score)

preds = xgb_model.predict(X_test)

print("f1 score=", f1_score(preds, y_test))

# we get an f1 score of arroud 0.64
    
print("ROC AUC score = ", roc_auc_score(y_test, preds))