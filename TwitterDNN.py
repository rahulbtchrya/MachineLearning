# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:55:37 2020

@author: rahul
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import re
from wordcloud import WordCloud
from gensim.parsing import remove_stopwords
import gensim
from wordcloud import STOPWORDS
#print(STOPWORDS)

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from nltk.stem import WordNetLemmatizer

def clean_data(text):    
    text = re.sub('@[\w]*', '', text)   # remove @user
    text = re.sub('&amp;','',text)             # remove &amp;
    text = re.sub('[?!.;:,,#@-]', '', text)  # remove special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text) # remove Unicode characters
    text = text.replace("[^A-Za-z]", "") # Replace everything except alphabets
    text = text.lower() # make everything lowercase for uniformity    
    # removing stop-words eg. 'we', 'our', 'ours', 'ourselves', 'just', 'don', "don't", 'should'
    text = remove_stopwords(text)    
    return text

# split each tweet into words, then lemmatize each word and rejoin them to a sentence
def lemmatize_text(text):
    words = text.split()
    lemm = WordNetLemmatizer()
    lemmatized_words = [lemm.lemmatize(word) for word in words]    
    return "".join(lemmatized_words)

    
def plot_model(history):
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    acc = hist['accuracy']
    loss = hist['loss']
    val_acc = hist['val_accuracy']
    val_loss = hist['val_loss']
    auc = hist['auc']
    val_auc = hist['val_auc']
    
    plt.figure()
    plt.plot(epochs,loss)
    plt.plot(epochs, val_loss)
    plt.legend(['loss', 'val_loss'])
    plt.show()
    
    plt.figure()
    plt.plot(epochs, acc, label='accuracy')
    plt.plot(epochs,val_acc, label='val_accuracy')
    plt.legend()
    plt.show()
    
    
    plt.figure()
    plt.plot(epochs, auc, label='AUC')
    plt.plot(epochs,val_auc, label='val_AUC')
    plt.legend()
    plt.show()
    


# this class will be used to reset the state of the model after every epoch
# this will be used along with the learning rate scheduler
class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self,epoch,logs):
        self.model.reset_states()
        

# *************************  read training data *******************************
df = pd.read_csv('train_E6oV3lV.csv')
print(df.head())

df.drop('id', axis=1, inplace=True)
df.drop_duplicates()
print(df.isna().sum())

# WAIT. DONT DO THIS. IT IS CAUSING THE F1 SCORE TO DROP FROM 65% TO 5%
# df = shuffle(df, random_state=101) # shuffle the dataset

tweets = df['tweet']
label = df['label']

tweets = tweets.apply(lambda x: clean_data(x))

# Now we normaize the words using lemmatization
# tweets = tweets.apply(lambda tweet: lemmatize_text(tweet))
# Note - we find that the scores decrese when we use lemmatization

# *********************** Exploratory data analysis ***************************

# lets generate word clouds for regular and racist tweets
wc = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate("".join([tweet for tweet in tweets]))
plt.figure(figsize=(10, 7)) 
plt.imshow(wc, interpolation="bilinear") 
plt.axis('off') 
plt.show()

# Lets find out the no. of words in a tweets
# split the tweets into individual words 
tweets_words = tweets.apply(lambda x : x.split())
#plot the count of no. of words in a tweet
len_tw = pd.Series([len(tweet) for tweet in tweets_words])
len_tw.hist(bins=10)

# from the graph we see that the maximum length of a tweet is 26, and average is 12

# convert the tweets to a list of sentences
tweets_list = tweets.tolist()
#print(tweets_list)

label = label.tolist()  # convert to a list
label = np.array(label)  # convert to numpy array


tweets_train, tweets_test, label_train, label_test = train_test_split(tweets_list, label, test_size=0.3, random_state=100,
                                                                      stratify=label)


METRICS = [
      # tf.keras.metrics.TruePositives(name='tp'),
      # tf.keras.metrics.FalsePositives(name='fp'),
      # tf.keras.metrics.TrueNegatives(name='tn'),
      # tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      # tf.keras.metrics.Precision(name='precision'),
      # tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
      ]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10, # wait till 10 epochs. can be increased later if needed
    mode='max',
    restore_best_weights=True)


# define the various parameters for the model
vocab_size = 1000  # take the 1000 most common words
embedding_dim = 16 # No. of features wanted eg. 16
max_length = 25   # max no. of words in a sentence

# define a tokenizer and train it on out list of words and sentences
tokenizer = Tokenizer(num_words=vocab_size , oov_token="<OOV>")
tokenizer.fit_on_texts(tweets_list)
word_index = tokenizer.word_index
print("Length of word index = ", len(word_index))
#print(tokenizer.word_index["pics"])

# convert the list of sentenses to tokenized list of words
sentences_train = tokenizer.texts_to_sequences(tweets_train)
sentences_train = pad_sequences(sentences_train, maxlen=max_length, padding='post', truncating='post')
print(sentences_train)
print(sentences_train.shape)

sentences_test = tokenizer.texts_to_sequences(tweets_test)
sentences_test = pad_sequences(sentences_test, maxlen=max_length, padding='post', truncating='post')

# ******* lets check if the data is skewed *****************************
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
neg_count = df['label'].value_counts()[0]
pos_count = df['label'].value_counts()[1]
total = pos_count + neg_count

# we see that there are only 7% instances postitve, so we define an initial bias
initial_bias = np.log([pos_count/neg_count])
# Define an output bias that would be applied to the output layer of the model
output_bias = tf.keras.initializers.Constant(initial_bias)

# calculate class weights
weight_for_0 = (1/neg_count) * (total)/2.0
weight_for_1 = (1/pos_count) * (total)/2.0
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))
# define the class-weight dictionary to be used for training the model
class_weight = {0:weight_for_0, 1:weight_for_1}

BATCH_SIZE=50
EPOCHS=20
LEARNING_RATE = 0.0001 # Default learning rate for the Adam optimizer is 0.001

# create a model
model = tf.keras.Sequential([
             tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
             tf.keras.layers.GlobalAveragePooling1D(),
             tf.keras.layers.Dense(6, activation='relu'),         
             tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), metrics=METRICS)
             

print("Training model....")
history = model.fit(sentences_train, label_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(sentences_test, label_test),
                    callbacks=[early_stopping],
                    class_weight=class_weight
                    )

plot_model(history)

preds = model.predict(x=sentences_test)
# The closer the prediction is to 1, the more positive the review is
pred_labels = [1 if pred>0.5 else 0 for pred in preds]

print("F1 score= ", f1_score(label_test,pred_labels))


# *************** Now we will use Subwords dataset with Bi-directional LSTM layers *********************
# define the subword tokenizer 
tokenizer_sub = tfds.features.text.SubwordTextEncoder.build_from_corpus(tweets_list, vocab_size, max_subword_length=5)
print("Vocab size is ", tokenizer_sub.vocab_size)

# Replace sentence data with encoded subwords
for i, tweet in enumerate(tweets_list):
    tweets_list[i] = tokenizer_sub.encode(tweet)
    
print(tweets_list[0])

sentences_sub = pad_sequences(tweets_list, maxlen=max_length,padding='post',truncating='post')

sentences_train_sub, sentences_test_sub, label_train, label_test = train_test_split(sentences_sub, label, 
                                                                            test_size=0.3, 
                                                                            random_state=100,
                                                                            stratify=label)


model_lstm = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), # note that flatten or  
                                                                                      # GlobalAveragePooling1D is not required
       
        # note that LSTM layer should have same no. of units the embedding dimention 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)), 
        # if we are using 2 LSTM layers, then the 1st layer should return a sequence for the 2nd layer                                                                    
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)    
    ])

model_lstm.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), metrics=METRICS)

history = model_lstm.fit(sentences_train_sub, label_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                         validation_data=(sentences_test_sub, label_test),
                         callbacks=[early_stopping],
                         class_weight=class_weight
                         )

plot_model(history)

preds = model_lstm.predict(x=sentences_test_sub)
# The closer the prediction is to 1, the more positive the review is
pred_labels = [1 if pred>0.5 else 0 for pred in preds]

print("F1 score= ", f1_score(label_test,pred_labels))


# *****************  now create and train a new lstm model with original sentences instead of subwords
model_lstm_new = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), # note that flatten or  
                                                                                      # GlobalAveragePooling1D is not required
       
        # note that LSTM layer should have same no. of units the embedding dimention 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)), 
        # if we are using 2 LSTM layers, then the 1st layer should return a sequence for the 2nd layer                                                                    
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)    
    ])

model_lstm_new.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), metrics=METRICS)

history = model_lstm_new.fit(sentences_train, label_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                             validation_data=(sentences_test, label_test),
                             callbacks=[early_stopping],
                             class_weight=class_weight
                             )

plot_model(history)

preds = model_lstm_new.predict(x=sentences_test)
# The closer the prediction is to 1, the more positive the review is
pred_labels = [1 if pred>0.5 else 0 for pred in preds]

print("F1 score= ", f1_score(label_test,pred_labels))

# We see the loss and AUC is just a little better on the subwords model than the regular word tokenization model


#************************* Now lets create and train a CNN model *********************************888
model_cnn = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(16,5,activation='relu'),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')    
    ])

model_cnn.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), metrics=METRICS)

history = model_cnn.fit(sentences_train, label_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                             validation_data=(sentences_test, label_test),
                             callbacks=[early_stopping],
                             class_weight=class_weight)

plot_model(history)

preds = model_cnn.predict(x=sentences_test)
# The closer the prediction is to 1, the more positive the review is
pred_labels = [1 if pred>0.5 else 0 for pred in preds]

print("F1 score= ", f1_score(label_test,pred_labels))

#************************* Now lets create and train a GRU model *********************************
model_gru = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(1, activation='sigmoid')    
    ])

model_gru.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), metrics=METRICS)

history = model_gru.fit(sentences_train, label_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                             validation_data=(sentences_test, label_test),
                             callbacks=[early_stopping],
                             class_weight=class_weight)

plot_model(history)

preds = model_gru.predict(x=sentences_test)
# The closer the prediction is to 1, the more positive the review is
pred_labels = [1 if pred>0.5 else 0 for pred in preds]

print("F1 score= ", f1_score(label_test,pred_labels))

#%%
# *********************************************************************************************************
# Considering the val_AUC parameter, the LSTM model with subwords-encoding seems to give the best performance. 
# We will try to optimize the lstm model using a learning-rate scheduler
# we will start with a very low learning rate of 1e-6 and after every 10 epochs we will multiply it by a power of 10
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10 **(epoch/10))
reset_states = ResetStatesCallback()

# define the model
model_new = None
model_new = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)), 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)    
    ])

model_new.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-6), metrics=METRICS)

history = model_new.fit(sentences_train_sub, label_train, epochs=100, batch_size=BATCH_SIZE,
                             callbacks=[lr_schedule
                                        # ,reset_states
                                        ],
                             class_weight=class_weight
                             )

plt.figure()
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-6, 10, 0, 1])
plt.show()

#%%

# from this graph we see that the best learning rate is between 1e-4 and 1e-3, so we would use 1e-4 
# earlier we were using 5e-4, but 1e-4 is giving beter resuts.
# Now we compile the model with the new learning rate and also save the best state of the model
model_final=None

model_final = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)), 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)    
    ])

model_final.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), metrics=METRICS)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint("my_checkpoint.h5", save_best_only=True)

history = model_final.fit(sentences_train_sub, label_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_data=(sentences_test_sub, label_test),
                             callbacks=[early_stopping, model_checkpoint
                                       # , reset_states
                                        ],
                             class_weight=class_weight
                             )


plot_model(history)
#%%
model=None
model = tf.keras.models.load_model("my_checkpoint.h5")

preds = model.predict(x=sentences_test_sub)
# The closer the prediction is to 1, the more positive the review is
pred_labels = [1 if pred>0.5 else 0 for pred in preds]

print("F1 score= ", f1_score(label_test, pred_labels))
#%%


# ********** for visualizing the emembeddings ****************

# First get the weights of the embedding layer
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io
# Write out the embedding vectors and metadata
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()