from keras.datasets import cifar10
from keras.utils import to_categorical

from src.parameters import CONSTANTS


def load_data():
    """
    Function to load the cifar10 dataset. Rather than downloading the files from the project's website and including
        them in the repo, I use the dataset dowload already available in Keras out of simplicity.

    Returns:
       Tuple containing the training data and labels, and the testing data and labels
    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    return x_train, y_train, x_test, y_test


def prepare_data(x_train, y_train, x_test, y_test):
    """
    Prepare cifar-10 data for training and testing by performing one hot encoding of class vectors, and normalizing the
        data between 0 and 1

    Returns:
       Tuple containing the normalized training data and encoded labels, and the normalized testing data and encoded labels
    """

    # One hot encoding of class vectors for neural network
    y_train = to_categorical(y_train, CONSTANTS['num_classes'])
    y_test = to_categorical(y_test, CONSTANTS['num_classes'])

    # Normalization of data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, y_train, x_test, y_test
