import os

from src.parameters import CONSTANTS, TRAINING_PARAMETERS


def train_model(model, x_train, y_train, x_test, y_test):
    """
    Trains defined model by using:
        - a batch size and number of epochs defined in the parameters file
        - shuffling of the training data before each epoch
    """

    model_history = model.fit(x_train, y_train,
                              batch_size=TRAINING_PARAMETERS['batch_size'],
                              epochs=TRAINING_PARAMETERS['epochs'],
                              validation_data=(x_test, y_test),
                              shuffle=True)

    return model, model_history


def save_model(model, model_name):
    """
    Saves model as .h5 file
    """

    model_path = os.path.join(CONSTANTS['saved_models_dir'], f"{model_name}.h5")
    model.save(model_path)
    print(f'Model saved at : {model_path}')


def evaluate_model(model, x_test, y_test):
    """
    Evaluates trained model's loss and accuracy on test dataset and prints results
    """

    loss, accuracy = model.evaluate(x_test, y_test)
    print('Model test loss:', loss)
    print('Model test accuracy:', accuracy)
