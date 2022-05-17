# import libraries 
import tensorflow as tf 
from tensorflow import keras
from keras import regularizers

# models
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization,
                                     Conv2D,
                                     MaxPool2D)


# optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

# utils
from keras.utils import np_utils

# scikit learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import cv2 
import os
import sys
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

import argparse

# let's load the data 
def load_dataset(net=True):
    """Utility function to load the FER2013 dataset.
    
    Code found and adjusted from: https://colab.research.google.com/github/RodolfoFerro/PyConCo20/blob/full-code/notebooks/Deep%20Learning%20Model.ipynb#scrollTo=9v3fuYQb139s
    
    It returns the formated tuples (X_train, y_train) , (X_test, y_test).

    Parameters
    ==========
    net : boolean
        This parameter is used to reshape the data from images in 
        (cols, rows, channels) format. In case that it is False, a standard
        format (cols, rows) is used.
    """

    # Load and filter in Training/not Training data:
    df = pd.read_csv(os.path.join("in", "fer2013.csv"))
    training = df.loc[df['Usage'] == 'Training']
    testing = df.loc[df['Usage'] != 'Training']

    # X_train values:
    X_train = training[['pixels']].values
    X_train = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_train]
    if net:
        X_train = [e.reshape((48, 48, 1)).astype('float32').flatten() for e in X_train] 
    else:
        X_train = [e.reshape((48, 48)) for e in X_train]
    X_train = np.array(X_train)

    # X_test values:
    X_test = testing[['pixels']].values
    X_test = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_test]
    if net:
        X_test = [e.reshape((48, 48, 1)).astype('float32').flatten() for e in X_test]
    else:
        X_test = [e.reshape((48, 48)) for e in X_test]
    X_test = np.array(X_test)

    # y_train values:
    y_train = training[['emotion']].values
    y_train = keras.utils.to_categorical(y_train)

    # y_test values
    y_test = testing[['emotion']].values
    y_test = keras.utils.to_categorical(y_test)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    label_names = ["angry",
         "disgust",
         "fear",
         "happy",
         "neutral",
         "sad",
         "surprise"]
    return X_train, y_train , X_test, y_test, label_names

# create model
def mdl(X_train, y_train, X_test, y_test):
    # sequential model
    model= tf.keras.models.Sequential()
    # add layers
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48,1)))
    model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) 
    model.add(Dense(256,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(512,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    # classification layer
    model.add(Dense(7, activation='softmax'))
    
    # compile model
    model.compile(
    optimizer = Adam(lr=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
  )
    
    # fit the training data to the model
    H = model.fit(X_train, y_train,
                    validation_data = (X_test, y_test),
                    epochs = 20,  
                    batch_size = 64)
   
    return model, H

# plotting function to see the loss and accuracy over time (epochs)
def plot_history(H):
    plt.style.use("seaborn-colorblind")

    # loss function plot
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    # accuracy plot
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    f.savefig(os.path.join("out", "emotion_loss_accuracy.jpg"))

# get predictions
def preds(model, X_test, y_test, label_names):
    predictions = model.predict(X_test, batch_size = 64)
    # get classification report
    report = (classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=label_names)) 
    # print report
    print(report)
    # create outpath
    p = os.path.join("out", "emotion_report.txt")
    # save classification report
    sys.stdout = open(p, "w")
    text_file = print(report)

def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-e", "-epochs", type=int, required=True, help="the emount of epochs of the model")
    args = vars(ap.parse_args())
    return args    
    
# let's get the code to run!
def main():
    #args = parse_args()
    X_train, y_train , X_test, y_test, label_names = load_dataset(False)
    model, H = mdl(X_train, y_train, X_test, y_test)
    plot_history(H)
    preds(model, X_test, y_test, label_names)

if __name__ == "__main__":
    main()

