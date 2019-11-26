from keras.datasets import cifar10
from keras.utils import to_categorical

from src.parameters import CONSTANTS


def load_data():
    """
    Function to load the cifar10 dataset. Rather than downloading the files from the project's website, I use the
        dataset already available in Keras out of simplicity.

    Returns:
       Tuple containing the training data and labels, and the testing data and labels
    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, CONSTANTS['num_classes'])
    y_test = to_categorical(y_test, CONSTANTS['num_classes'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test)
