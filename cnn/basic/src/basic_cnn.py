'''
Simple Convolutional Neural Network (CNN) example in Keras (Python)
The program performs classification of input images into two groups: hand-drawn circles and hand-drawn lines
Author: Goran Trlin
Find more tutorials and code samples on:
https://playandlearntocode.com
'''

import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras import backend

from classes.extraction.folder_helper import FolderHelper
from classes.extraction.image_helper import ImageHelper

print('CNN example starting....')
if backend.image_data_format() != 'channels_last':
    raise Exception('This example expects a channels_last backend,i.e. TensorFlow')

# image properties:
image_height = 16
image_width = 16
number_of_channels = 3  # r,g,b channels of the input images
train_samples = 40

# input samples table format:
input_object_shape = (image_height, image_width, number_of_channels)  # expected Channels-last backend (Tensor Flow)

# load image data here (inputs):
fh = FolderHelper()
input_images = fh.process_images_folder('../images', 16, 16)
x_training = input_images

# load target labels here (target outputs):
labels = fh.process_labels_file('../data/labels.txt')
y_training = np.array(labels['correct_answer'])

# some preprocessing
x_training = x_training.astype('float32')
x_training /= 255.0

# convert to one-hot vectors:
y_training = keras.utils.to_categorical(y_training, num_classes=2)  # expects 0,1 values!

# build the CNN model:
model = Sequential()
layers = [
    # 1st convolutional layer
    Conv2D(16, kernel_size=(4, 4), activation='relu', input_shape=input_object_shape),
    # 2nd convolutional layer
    Conv2D(128, kernel_size=(8, 8), activation='relu'),
    # max pooling:
    MaxPooling2D(pool_size=(4, 4)),
    # some regularization:
    Dropout(0.1),
    # adjust the data object shape:
    Flatten(),
    # output layer:
    Dense(2, activation='softmax')
]

# attach layers to the model:
for layer in layers:
    model.add(layer)

# compile the model:
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

# set hyperparameters and train the model:
model.fit(
    x_training[:train_samples], y_training[:train_samples],
    batch_size=1,
    epochs=100,
    verbose=1,
    validation_split=0.2,  # split input data in 80:20
)

# make predictions:
num_of_prediction_samples = 40

pred = model.predict(x_training[:num_of_prediction_samples])

# find correspoding labels:
pred = np.argmax(pred, axis=1)
label = np.argmax(y_training[:num_of_prediction_samples], axis=1)

print('Predicted labels:')
print(pred)
print('True labels:')
print(label)

print('CNN example completed!')
