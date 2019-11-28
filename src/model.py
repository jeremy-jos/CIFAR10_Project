from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from src.parameters import CONSTANTS


def classifier_model(x_train):
    """
    Define structure of CNN classifier model. The structure I coded is a sequential model with:
        - two 2D Convolutional layer that use:
            - Rectified Linear Unit (ReLU) activation
            - He uniform variance scaling initializer
        - a 2D Polling Layer that downscale the image by 2 horizontally and vertically
        - a Flatten layer to flatten the intput for the following Dense layer
        - a Dense layer which is a usual fully connected neural network layer

    """

    model = Sequential()
    model.add(Conv2D(
        32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_train.shape[1:])
    )
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(CONSTANTS['num_classes'], activation='softmax'))

    return model
