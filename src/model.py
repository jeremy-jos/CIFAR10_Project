from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from src.parameters import CONSTANTS, TRAINING_PARAMETERS


def classifier_model():
    """
    Defines structure of CNN classifier model.

    The structure I coded is a sequential model inspired by the VGG models that is made up of 2 blocks, each block containing:
        - two 2D Convolutional layer that use:
            - 32 and 64 filters for the first and second blocks
            - (3,3) size filters
            - Rectified Linear Unit (ReLU) activation
            - He uniform variance scaling initializer
            - padding to be sure that the shape of the outputs matches that of the inputs
        - a 2D Polling Layer that downscales the image by 2 horizontally and vertically
        - a Dropout Layer to add some regularization to the model. I added more dropout to the second block than to the first

    These blocks are followed by a classifier that is made of:
        - a Flatten layer to flatten the input for the following Dense layer
        - a first Dense layer which is a usual fully connected neural network layer with ReLU activation
        - a final Dense layer that generates predictions with a softmax activation

    The model is compiled with :
        - a categorical crossentropy loss given the multiclass task
        - a Stochastic Gradient Descent optimizer
        - an accuracy metric
    """

    # Sequential Model
    model = Sequential()

    # First Block
    model.add(Conv2D(
        32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=CONSTANTS['input_shape'])
    )
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Second Block
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.35))

    # Classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(CONSTANTS['num_classes'], activation='softmax'))

    # Compile Model using Stochastic Gradient Descent Optimizer
    opt = SGD(
        lr=TRAINING_PARAMETERS['sgd_learning_rate'],
        momentum=TRAINING_PARAMETERS['sgd_momentum']
    )

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
