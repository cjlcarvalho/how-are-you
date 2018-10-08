import cv2
import numpy as np
import os

from keras.utils.np_utils import to_categorical

from ann.model import model

def extract_faces(img):
    cropped_images = []
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        cropped_images.append(roi)
    return cropped_images
            

def X(files):

    result = []

    for f in files:
        if f.endswith('.jpg'):
            img = cv2.imread('images/' + f)
            faces = extract_faces(img)
            for face in faces:
                result.append(cv2.resize(face, (96, 96)))

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

    m.fit(x_train, y_train, batch_size=100, epochs=300, validation_data=(x_test, y_test))

    m.save_weights('weights/cnn_emotions.weights')
    m.save('weights/cnn_emotions_caio.model')
