from keras.datasets import cifar10


def load_data():
    """
    Function to load the cifar10 dataset. Rather than downloading the files from the project's website, I use the
        dataset already available in Keras out of simplicity.

    Returns:
       Tuple containing the training data and labels, and the testing data and labels
    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    return (x_train, y_train), (x_test, y_test)

