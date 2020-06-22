# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:47:25 2020

@author: rahul

AI Crowd - Food Recognition

"""
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#import tensorflow_docs as tfdocs - need to install


from sklearn import preprocessing
from sklearn.metrics import f1_score, classification_report 



LEARNING_RATE = 0.0001
BATCH_SIZE = 200
EPOCHS = 10

AUTOTUNE = tf.data.experimental.AUTOTUNE

def createModel(my_learning_rate):
    # Load the inception V3 model
    local_weights_file='.\InceptionV3ModelWeights\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    pretrained_inception_model = InceptionV3(input_shape=(150,150,3), include_top=False, weights=None)
    pretrained_inception_model.load_weights(local_weights_file)
    # By specifying the include_top=False argument, we load a network that doesn't include the classification layers at the top—ideal for feature extraction.
    
    # Make the base model non-trainable
    for layer in pretrained_inception_model.layers:
        layer.trainable=False
        
    # Get the mixed7 layer from the inception model and get its output
    mixed7layer = pretrained_inception_model.get_layer('mixed7')
    output = mixed7layer.output
    
    # Add our costom layers on top of the pre-trained model
    x = layers.Flatten() (output)
    x = layers.Dense(1024, activation='relu') (x)
    x = layers.Dropout(0.2) (x)
    x = layers.Dense(61, activation='softmax') (x)  # We hve 61 distinct classes hence 61 logits
        
    # Create the model
    model = Model(pretrained_inception_model.input,x)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])
                          
    return model



# read an image file and convert it into a 3d array of shape(150,150,3)   
def getImageDataset(imageId):
    img_path = '.\\FoodRecognition\\train_images\\' + imageId
    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
    #img.show()
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    #x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3). This is needed for a single image, not for a batch
    x /= 255  # Rescale by 1/255
    #print(x.shape)
    return x



# using this to convert our series objects to numpy arrays. The to_numpy() function is not
# working for some reason
def convertToNumpyArray(series):
    tmp = []
    for itm in series:
        tmp.append(itm)
    return np.array(tmp)



#@title Define the plotting function
def plot_curve(epochs, hist, list_of_metrics_acc, list_of_metrics_loss):
  """Plot a curve of one or more classification metrics vs. epoch."""  
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")
  plt.title('Training and validation accuracy')
  for m in list_of_metrics_acc:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)
  plt.legend()

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")
  plt.title('Training and validation loss')
  for m in list_of_metrics_loss:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)
  plt.legend()



# one-hot encoding for label - Not needed here since we are using label encoding
# def getLabel(label_name):
#     return label_name == CLASS_NAMES


train_df = pd.read_csv('.\\FoodRecognition\\train.csv')

#rain_df_temp = train_df[1:10]


# Label encode the class_name values
print("Endoing labels....")
le = preprocessing.LabelEncoder()
ClassLabels = le.fit_transform(train_df['ClassName'])
train_df['ClassName'] = ClassLabels

# Apply this function to each element of the series
print("Convering images....")
feature_ds = train_df['ImageId'].map(getImageDataset)
#print(feature)
label_ds = train_df.pop('ClassName')


# Split the training data into train and test sets - 70% train and 30% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature_ds, label_ds, test_size=0.3)


# The feature and label both are series objects. We need to convert them to numpy arrays to train the model
print("Converting to numpy....")
feature = convertToNumpyArray(X_train)
label = convertToNumpyArray(y_train)


#%%


model = None # discard any pre-existing models
model = createModel(my_learning_rate=LEARNING_RATE)
print("Model loaded")

# define list of call-back functions
callbacks_list = [
  #  tfdocs.modeling.EpochDots(), #prints a . for each epoch, and a full set of metrics every 100 epochs.
    tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(".\\FoodRecognition\\logs\\")
    ]

print("Begin training.....")
history = model.fit(x=feature,
                    y=label, 
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=callbacks_list
                    )

#%%
epochs = history.epoch
hist = pd.DataFrame(history.history)

# Plot a graph of the metric vs. epochs.
print("Plotting training graphs...")
list_of_metrics_acc = ['accuracy', 'val_accuracy']
list_of_metrics_loss = ['loss', 'val_loss']
plot_curve(epochs, hist, list_of_metrics_acc, list_of_metrics_loss)
                    
#%%
# Evaluate the model
print("Evaluate the model....")
feature_test = convertToNumpyArray(X_test)
label_test = convertToNumpyArray(y_test)
# test_loss, test_acc = model.evaluate(feature_test, label_test, batch_size=1)
results = model.evaluate(feature_test, label_test, batch_size=BATCH_SIZE)

#%%
# Make predictions
train_df_org = pd.read_csv('.\\FoodRecognition\\train.csv')
predict_df = train_df_org[1000:2000]
predict_X = convertToNumpyArray(predict_df['ImageId'].map(getImageDataset))
predictions = model.predict(x=predict_X)
#print(predictions)

predicted_indexes = []

for pred in predictions:
    predicted_indexes.append(np.argmax(pred))

predicted_labels = le.inverse_transform(predicted_indexes) 
#print(predicted_labels)


#%%
actual_labels = le.fit_transform(predict_df['ClassName'])
#print("F1 score :", f1_score(predict_df['ClassName'], predicted_labels, average='micro'))
print("F1 score :", f1_score(actual_labels, predicted_indexes, average='micro'))

# print classification report
print(classification_report(predict_df['ClassName'],predicted_labels))

#%%
# plot the predictions - not much help
plt.figure()
plt.xlabel("ImageId")
plt.ylabel("Classname")
plt.scatter(predict_df['ImageId'], actual_labels)
plt.scatter(predict_df['ImageId'], predicted_indexes)
plt.legend()

#%%
# d = {'ImageId': predict_df['ImageId'] , 'ClassName': predicted_labels}
# df_preds = pd.DataFrame(d)

df_preds = pd.DataFrame(predicted_labels, columns=['ClassName'])
df_preds.to_csv('submission.csv', index=False)


# #%%
# list_ds = tf.data.Dataset.list_files(".\\FoodRecognition\\train_images\\train_images\\*")
    
# for f in list_ds.take(5):
#   print(f.numpy())

