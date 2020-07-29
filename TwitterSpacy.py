# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 17:42:01 2020

@author: rahul

Here we are doing twitter hate-speech analysis using Spacy Text Classifier

"""

import pandas as pd
import re
from gensim.parsing import remove_stopwords
import spacy
from spacy.util import minibatch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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

tweets = df['tweet']
labels = df['label']

# clean the texts
tweets = tweets.apply(lambda x : clean_data(x))

tweets_train, tweets_test, labels_train, labels_test = train_test_split(tweets, labels, 
                                                                        test_size=0.3,
                                                                        random_state=1,
                                                                        stratify=labels)

# Create a blank spacy model
nlp = spacy.blank("en")

# Create the TextCategorizer with exclusive classes and "cnn" architecture
textcat = nlp.create_pipe("textcat", config={
                                        "exclusive_classes": True,
                                        "architecture": "simple_cnn"
                                        })

# Note: The spacy text categorizer supports only CNN, BOW(bag of words) and an ensemble of the both
# To use other methods use word vectors from large models along with algorithms like XGBoost. 
# Refer to TwitterSpacyVec.py for more details

textcat.add_label("POSITIVE")
textcat.add_label("NEGETIVE")

nlp.add_pipe(textcat, last=True)

# prepare the training data
texts_train = tweets_train.values
labels_cats = [{'cats': {"POSITIVE": not bool(label), "NEGETIVE": bool(label)}} for label in labels_train] # 0=positive,1=negetive

data = list(zip(texts_train,labels_cats))

print(data[:1])


# ***************  train the model ********************************************

spacy.util.fix_random_seed(1)

optimizer = nlp.begin_training()

losses = {}

for epoch in range(2):
    random.shuffle(data)
    batches = minibatch(data,size=10)
    for batch in batches:
        texts,labels = zip(*batch)
        nlp.update(texts, labels, sgd=optimizer, losses=losses)
    print(losses)
    

# making predictions
text = "This tea cup was full of holes. Do not recommend."
doc = nlp(text)
print(doc.cats)


# *****************  evaluate on test set  ************************************

# convert test tweets to list of nlp docs
test_docs = list(nlp.pipe(tweets_test))

# get the text-categorizer pipe
textcat = nlp.get_pipe('textcat')

scores, _ = textcat.predict(test_docs)

predicted_clases = scores.argmax(axis=1)

correct_predictions = predicted_clases==labels_test

accuracy = correct_predictions.mean()

print("Accuracy=", accuracy)

print("F1 score=", f1_score(predicted_clases,labels_test))

# we get an f1 score of arroud 0.66