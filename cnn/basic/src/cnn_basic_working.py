import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense,Dropout, Flatten, MaxPooling2D
from keras import backend

print('Data format from the backend:')
print(backend.image_data_format())

image_height = 16
image_width = 16
number_of_channels = 3
input_object_shape = (image_height, image_width, number_of_channels)  # expected Channels-last backend,ie tensorflow

from classes.extraction.folder_helper import FolderHelper
from classes.extraction.image_helper import ImageHelper

# Load image data here (inputs):

fh = FolderHelper()
input_images = fh.process_images_folder('../images', 16, 16)
x_train = input_images
print(x_train.shape)

# Load target labels here (target outputs):
labels = fh.process_labels_file('../data/labels.txt')
y_train = np.array(labels['correct_answer'])
print(y_train.shape)

x_train = x_train.astype('float32')
x_train /= 255

y_train = keras.utils.to_categorical(y_train, num_classes=2)  # expects 0,1 values!
print(y_train)
# Build the CNN model:
model = Sequential()
layers = [
    # 1st convolutional layer with 16 kernels, with each kernel being a 3x3 box
    Conv2D(8, kernel_size=(8, 8), activation='relu', input_shape=input_object_shape),
    # 2nd convolutional layer with 32 kernels, with each kernel being a 3x3 box
    Conv2D(16, kernel_size=(4, 4), activation='relu'),
    # Max pooling, preparing for output format:
    MaxPooling2D(pool_size=(2, 2)),
    #Dropout(0.5),
    Flatten(),  # adjust the data object shape
    #Dense(4, activation='relu'),  # standard dense layer
    Dense(2, activation='softmax')  # output layer
]

for layer in layers:
    model.add(layer)

# Compile the model:
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=['accuracy'])

train_samples = 40
# Train the model:
model.fit(
    x_train[:train_samples], y_train[:train_samples],
    batch_size=1,
    epochs=50,
    verbose=1,
    validation_data = (x_train[train_samples:], y_train[train_samples:])
)

# Make a prediction:
num_of_test_samples = 10

pred = model.predict(x_train[:num_of_test_samples])
print(pred)
pred = np.argmax(pred,axis=1)
label = np.argmax(y_train[:num_of_test_samples],axis=1)

print('final:')
print(pred)
print(label)

print('Done')
