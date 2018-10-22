import cv2
import numpy as np
import os

from keras.utils.np_utils import to_categorical

from ann.model import model
from ann.utils import extract_face

def generate_data(files, classes):

    print('Generating X and Y...')

    X = []
    Y = []

    for f in files:

        if f.lower().endswith('.jpg'):

            img = cv2.imread('images/' + f)
            face = extract_face(img)

            if face is not None:

                X.append(cv2.resize(face, (96, 96)))

                for i in range(len(classes)):

                    if classes[i] in f:

                        Y.append(i)

    return (np.array(X), np.array(Y))

def train(classes):

    files = os.listdir('images')

    x, y = generate_data(files, classes)

    print('Found %d data to evaluate' % len(x))

    for face in x:

        cv2.imshow('Face', face)

        cv2.waitKey(0)

    y = to_categorical(y)

    seventy_five_percent = int(len(files) * 0.75)

    x_train = x[0:seventy_five_percent, :, :, :]
    x_test = x[seventy_five_percent:, :, :, :]

    y_train = y[0:seventy_five_percent, :]
    y_test = y[seventy_five_percent:, :]

    m = model(len(classes))

    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    m.fit(x_train, y_train, batch_size=100, epochs=100, validation_data=(x_test, y_test))

    m.save_weights('weights/cnn_emotions.weights')
    m.save('weights/cnn_emotions.model')
