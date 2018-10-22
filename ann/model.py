from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model

def model(num_classes):

    m = Sequential()

    m.add(Conv2D(10, (5, 5), activation='relu', input_shape=(48, 48, 3)))
    m.add(MaxPooling2D((2,2)))

    m.add(Conv2D(10, (5, 5), activation='relu'))
    m.add(MaxPooling2D((2,2)))

    m.add(Conv2D(10, (3, 3), activation='relu'))
    m.add(MaxPooling2D((2,2)))

    m.add(Flatten())
    m.add(Dense(256, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(64, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))

    return m

def plot(model, file_path):

    plot_model(model, to_file=file_path, show_shapes=True)
