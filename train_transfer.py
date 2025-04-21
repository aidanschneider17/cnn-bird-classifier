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

imgsize = 224

label_encoder = {}
label_decoder = {}

base_dir = '/data/csc6621/24-summer-team-E/'
train_image_paths = []
valid_image_paths = []
y_train = []
y_valid = []

def get_image_paths(filepath):
    global train_image_paths
    global valid_image_paths
    global y_train
    global y_valid

    filepath = os.path.join(base_dir, filepath)

    if filepath.split('/')[-3] == 'train':
        train_image_paths.append(filepath)
        y_train.append(label_encoder[filepath.split('/')[-2]])
    elif filepath.split('/')[-3] == 'valid':
        valid_image_paths.append(filepath)
        y_valid.append(label_encoder[filepath.split('/')[-2]])

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [imgsize, imgsize])
    image = image / 255.0
    return image

def main():
    global train_image_paths
    global valid_image_paths
    global y_train
    global y_valid
    global label_encoder
    global label_decoder

    batch_size = 128

    base_dir = '/data/csc6621/24-summer-team-E/'
    df_csv = pd.read_csv(base_dir + 'birds.csv')

    index = 0
    folder_path = os.path.join(base_dir, 'train')

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            label_encoder[item] = index
            label_decoder[index] = item
            index += 1

    df_csv['filepaths'].apply(get_image_paths)

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

    for images, labels in train_dataset.take(3):  # Take the first 3 batches
        print("Image batch shape:", images.shape)
        print("Labels batch shape:", labels.shape)
        print("A few label examples:", labels[:5])


    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(525, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=10, validation_data=valid_dataset)

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    ft_history = model.fit(train_dataset, epochs=5, validation_data=valid_dataset)

    model.save('transfer_model.keras')

    with open('transfer_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    with open('transfer_history_ft.pkl', 'wb') as file:
        pickle.dump(ft_history.history, file)

    model.summary()

if __name__ == '__main__':
    main()

