# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:12:38 2020

@author: rahul
"""

#%%
# *********** Tokenization ***************************
import nltk
#nltk.download()

from nltk.tokenize import sent_tokenize, word_tokenize

text = "Hi John, how are you doing? I will be travelling to your city. Lets catchup."

sentences = sent_tokenize(text)
print(sentences)

words = word_tokenize(text)               
print(words)

#%%
# ********* Normalization ******************************
from nltk.stem import PorterStemmer, WordNetLemmatizer                      

stemmer = PorterStemmer()
print(stemmer.stem("playing"))
print(stemmer.stem("plays"))
print(stemmer.stem("increases"))

lem = WordNetLemmatizer()
print(lem.lemmatize("increases"))
print(lem.lemmatize("running"))
print(lem.lemmatize("running", pos="v"))  # parts of peach tagging for verb

#%%
# ************ Parts of speech tagging ********************
from nltk import pos_tag

text = "Hi John, how are you doing? I will be travelling to your city. Lets catchup."

tokens = word_tokenize(text)

tags = pos_tag(tokens)

print(tags)

#%%
#************ Synonyms ************************************
from nltk.corpus import wordnet

synonyms = wordnet.synsets('good')
print(synonyms) 

#%%
#********** Obtaining n-grams ******************************
from nltk import ngrams

sentence = "I love to play football and cricket" 

bi_grams = ngrams(word_tokenize(sentence), 2) # 2 for bi-grams

for gram in bi_grams:
    print(gram)
    
#%%
# Topic modelling using Latent Dirichlet Allocation (LDA)
# https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/

import gensim
from gensim import corpora

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father." 
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."

doc_complete = [doc1, doc2, doc3]
doc_clean = [doc.split() for doc in doc_complete]

# Creating the term dictionary of our corpus, where every unique term is assigned an index.   
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above. 
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix
lda_model = Lda(corpus=doc_term_matrix, num_topics=3, id2word=dictionary,passes=50)

# Results
print(lda_model.print_topics())
