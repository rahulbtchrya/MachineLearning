## Introduction

This is my repository of Machine Learning and NLP code files. Majority of the python files are related to NLP, although there are some which cater to different areas as well. 

## Python code files

Below is the description of the various code files:

- `TwitterNLTK.py` - This code file uses NLTK and Gensim libraries to classify twitter text tweets into racist or non-racist. We use a combination of Word2Vec and XGBoost to create and train the model.
- `TwitterTextBlob.py` - This code file uses the TextBlob library to classify twitter text tweets into racist or non-racist.
- `TwitterDNN.py` - This code file uses a model based on Deep Neural Networks, specifically Bi-directional LSTM cells to classify twitter text tweets into racist or non-racist.
- `TwitterSpacy.py` - This code demonstrates the use of Spacy's built-in text classifier to classify twitter texts.
- `TwitterSpacyVec.py` - This code uses a combination of Spacy's document vectors and XGBoost for classification of tweets. The vectors are obtained from Spacy's large english model "en_core_web_lg"
- `FoodRecognition.py` - This code uses Google's Inception V3 pre-trained model to classify images of food items.
- `DimentionalityReduction.py` - In this code we use common dimentionality reduction techniques on the auto-insurance dataset. We start with data vizualization and cleaning. Then we perform imputaion of missing values followed by normalization. Then we reduce our features using statictical techniques, like removing highly correlated features, removing variables having very low logistic regression coefficients, and finally implementing the random-forest classifier to determine the top 10 most important features.
- `FakeNewsSpacy.py` - Here we are using Spacy's built-in text categorizer to create and train a model to classify a news text as fake or real. 

