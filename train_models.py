import os
from scipy.io import loadmat
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from sklearn.svm import SVC
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input,LSTM,SimpleRNN
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as pltj
import keras
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model
from tensorflow.keras.layers import SimpleRNN, Dense
import csv
import pandas as pd
from sklearn.metrics import classification_report, f1_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Conv1D,MaxPooling1D, AveragePooling1D,Dropout
from scipy.signal import butter, lfilter
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle

label_encoder = {}
label_decoder = {}

x_train = []
x_test = []
x_val = []
y_train = []
y_test = []
y_val = []

imgsize = 224
batch_size = 128

base_dir = '/data/csc6621/24-summer-team-E/'

def main():

    global x_train
    global x_test
    global x_val
    global y_train
    global y_test
    global y_val

    global imgsize

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, 'train'),
        image_size=(imgsize, imgsize),
        batch_size=batch_size,
        shuffle=True,
        label_mode='categorical'
    )

    valid_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(base_dir, 'valid'),
        image_size=(imgsize, imgsize),
        batch_size=batch_size,
        shuffle=True,
        label_mode='categorical'
    )


    cnn_model = tf.keras.Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(imgsize, imgsize, 3)),
        Conv2D(16, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.2),

        Conv2D(32, (3, 3), activation='relu', input_shape=(imgsize, imgsize, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.2),

        Conv2D(64, (5, 5), activation='relu', input_shape=(imgsize, imgsize, 3)),
        Conv2D(64, (5, 5), strides=2, activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2), strides=2),
        Dropout(0.2),

        Flatten(),
        Dense(2048, activation='relu'),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(525, activation='softmax')
     ])

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    history = cnn_model.fit(train_dataset, epochs=20, validation_data=valid_dataset)
    cnn_model.save('cnn_model.keras')

    with open('cnn_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)


    cnn_model.summary()


if __name__ == '__main__':
    main()

