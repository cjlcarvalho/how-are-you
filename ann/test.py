import cv2
import numpy as np

from ann.model import model
from ann.utils import extract_face

def test(image, classes):

    face = extract_face(image)

    if face is not None:

        face = cv2.resize(face, (96, 96))

        # It will expect a 4D array as input, because of x_train
        # So put the image inside of a np array
        face = np.array([face])

        m = model(len(classes))
        m.load_weights('weights/cnn_emotions.weights')
        m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        prediction = m.predict(face)

        best = np.argsort(prediction)[0, :]

        cv2.putText(image, classes[best[-1]], (image.shape[1] - 100, image.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

        return image

    print('Face not found')

    return None

