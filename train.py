import cv2
import numpy as np
import os

from keras.utils.np_utils import to_categorical

from model import model

def X(files):

    result = []

    for f in files:

        if f.endswith('.jpg'):

            result.append(cv2.resize(cv2.imread('images/' + f), (256, 256)))

    return np.array(result)

def Y(files, classes):

    result = []

    for f in files:

        for i in range(len(classes)):

            if classes[i] in f:

                result.append(i)

    return np.array(result)

def train(classes):

    files = os.listdir('images')

    x = X(files)
    y = Y(files, classes)

    y = to_categorical(y)

    seventy_five_percent = int(len(files) * 0.75)

    x_train = x[0:seventy_five_percent, :, :, :]
    x_test = x[seventy_five_percent:, :, :, :]

    y_train = y[0:seventy_five_percent, :]
    y_test = y[seventy_five_percent:, :]

    m = model(len(classes))

    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    m.fit(x_train, y_train, batch_size=100, epochs=300, validation_split=0.1)

    m.save_weights('cnn_emotions.weights')
    m.save('cnn_emotions_caio.model')
