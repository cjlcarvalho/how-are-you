import cv2
import numpy as np
import os

from keras.utils.np_utils import to_categorical

from ann.model import model


def extract_faces(img):
    # Reading cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Converting RGB to Gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If there is no detected face, return None
    if not len(faces) > 0:
        return

    i = 0
    biggest_index = 0
    largest_area = 0
    
    for (x, y, w, h) in faces:
        if (w * h) > largest_area:
            biggest_index = i
            largest_area = w * h
        i = i + 1

    # Unpack values
    x = faces[biggest_index, 0]
    y = faces[biggest_index, 1]
    w = faces[biggest_index, 2]
    h = faces[biggest_index, 3]

    # Return biggest face area
    roi = img[y:y + h, x:x + w]
    return roi


def X(files):
    result = []

    for f in files:
        # Make it lowercase first
        # JPG and jpg is not the same
        if f.lower().endswith('.jpg'):
            img = cv2.imread('images/' + f)
            face = extract_faces(img)
            if face is not None:
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
