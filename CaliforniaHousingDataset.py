# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:17:57 2020

@author: rahul
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

train_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# Normalize the data
train_df_norm = (train_df - train_df.mean() ) / train_df.std()
test_df_norm = (test_df - test_df.mean()) / test_df.std()


# shuffle the training dataset
train_df_norm = train_df.reindex(np.random.permutation(train_df.index))

# define model
def create_model(feature_columns,learning_rate):
    
    numeric_feature_names = ["total_rooms","total_bedrooms"]

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.DenseFeatures(feature_columns)) # feature layer
    #model.add(tf.keras.layers.DenseFeatures([tf.feature_column.numeric_column(col_name) for col_name in numeric_feature_names]))
    model.add(tf.keras.layers.Dense(units=20, activation='relu', name='hidden1', kernel_regularizer=tf.keras.regularizers.l2(0.04)))
    model.add(tf.keras.layers.Dense(units=12, activation='relu', name='hidden2', kernel_regularizer=tf.keras.regularizers.l2(0.04)))
    model.add(tf.keras.layers.Dense(units=1, name='output'))

    #   This is when you are not using DNN 
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),loss=tf.losses.mean_squared_error,
    #               metrics=[tf.keras.metrics.RootMeanSquaredError()])  
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
                  loss="mean_squared_error", 
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    
    return model


# train model
def train_model(model,df,label_name,batch_size,num_epochs, my_validation_split):
    
    features = {key:np.array(val) for key,val in df.items()}
    label = np.array(features.pop(label_name))
    
    history = model.fit(x=features, 
                        y=label, 
                        batch_size=batch_size,
                        epochs=num_epochs
                       # validation_split=my_validation_split - doesn't work
                        )
    trained_weight = model.get_weights()[0]
    trained_bais = model.get_weights()[1]    
    epochs = history.epoch  
    hist = pd.DataFrame(history.history)  
    mse = hist["mean_squared_error"]
    
    return trained_weight,trained_bais,epochs,mse


# plot the model (for a single feature)
def plot_the_model(trained_weight,trained_bais,feature_name,label_name):
    plt.xlabel(feature_name)
    plt.ylabel(label_name)
    
    random_examples = train_df.sample(n=200)
    plt.scatter(random_examples[feature_name],random_examples[label_name])
    
    x0 = 0
    y0 = trained_bais
    x1 = 10000
    y1 = trained_weight * x1 + trained_bais
    plt.plot([x0,x1],[y0,y1], c='r')    
    plt.show()
    
    
# plot the loss
def plot_the_loss_curve(epochs, mse):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.plot(epochs,mse,label="Loss")
    plt.legend()
    plt.ylim([mse.min() * 0.95, mse.max() * 1.03])
    plt.show()
    
    
    
#feature_names = ["total_rooms","total_bedrooms"]
label_name = "median_house_value"


numeric_feature_names = ["total_rooms","total_bedrooms"]
categorical_feature_names = []  # doesn't have any categorical feature yet

numeric_feature_columns = [tf.feature_column.numeric_column(col_name) for col_name in numeric_feature_names]

categorical_feature_columns = [tf.feature_column.indicator_column(
                              tf.feature_column.categorical_column_with_vocabulary_list(col_name,
                                                                               vocabulary_list=train_df_norm[col_name].unique())) 
                              for col_name in categorical_feature_names]

my_feature_columns = numeric_feature_columns + categorical_feature_columns

my_model = None
my_model = create_model(feature_columns=my_feature_columns, learning_rate=0.01)

weight,bais,epochs,mse = train_model(my_model, train_df, label_name, 
                                      batch_size=1000, 
                                      num_epochs=20,
                                      my_validation_split=0.3  # doesn't work, not used
                                      )

#print("Trained weight = {} and trained bias = {}".format(weight,bais))

print("Training complete")

#plot_the_model(weight,bais,feature_names,label_name)

plot_the_loss_curve(epochs,mse)

# Test the model
test_features = {key:np.array(val) for key,val in test_df.items()}
test_label = np.array(test_features.pop(label_name))

print("Test data results :- ")
my_model.evaluate(x=test_features, y=test_label , batch_size=1000)    
        


