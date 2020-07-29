# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 21:27:51 2020

@author: rahul

In this program we will use WordBlob to classify tweets as rasist or not 

"""

import pandas as pd
from gensim.parsing import remove_stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from textblob import TextBlob, Word
from textblob import classifiers
import random

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

df = df[:5000] # using the whole dataset hangs the system

tweets = df['tweet']
labels = df['label']

# clean the data set
tweets = tweets.apply(lambda x : clean_data(x))

tweets = tweets.tolist()
labels = labels.tolist()

#******************************************************************************
# lets try to find out what a summary of the tweets. For that, we would pick 
# some tweets and find out the nouns in them

blob = TextBlob("".join(tweets))
nouns = []
for word,tag in blob.tags:
    if tag == 'NN':
        nouns.append(word)
print("This text is about:")
for item in random.sample(nouns,5):
    word = Word(item)
    print(word)


# *****************************************************************************
tweets_train, tweets_test, labels_train, labels_test = train_test_split(tweets, labels, test_size=0.3, random_state=100,
                                                                      stratify=labels)


# We need to convert the train and test data into a list of tuples of the type (tweet,label)
training_corpus = list(zip(tweets_train,labels_train))
test_corpus = list(zip(tweets_test,labels_test))

print("Training classifier......")
classifier = classifiers.DecisionTreeClassifier(training_corpus)

print("accuracy=", classifier.accuracy(test_corpus))


predictions = []
for tweet in tweets_test:
    pred = classifier.classify(tweet)
    predictions.append(pred)
    
print("F1 score=" , f1_score(labels_test,predictions))

# we get F1 score=0.4662576687116564 for 5000 docs