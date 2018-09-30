import cv2
import numpy as np

from ann.model import model

def test(image_path, classes):

    im = cv2.imread(image_path)

    image = cv2.resize(im, (96, 96))

    # It will expect a 4D array as input, because of x_train
    # So put the image inside of a np array
    image = np.array([image])

    m = model(len(classes))
    m.load_weights('weights/cnn_emotions.weights')
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    prediction = m.predict(image)

    best = np.argsort(prediction)[0, :]

    cv2.putText(im, classes[best[-1]], (im.shape[1] - 100, im.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    cv2.imshow('Prediction', im)

    if cv2.waitKey(0) & 0xFF == ord('q'):

        return
