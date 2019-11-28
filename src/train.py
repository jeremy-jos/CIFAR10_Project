import os

from keras.optimizers import SGD

from src.parameters import CONSTANTS, TRAINING_PARAMETERS


def train_model(model, x_train, y_train, x_test, y_test):
    """
    Trains defined model by using:
        - Stochastic Gradient Descent Optimizer with learning rate and momentum defined in parameters file
        - a categorical crossentropy loss
        - an accuracy metric
        - a batch size and number of epochs defined in the parameters file
        - shuffling the training data before each epoch
    """

    # Define Stochastic Gradient Descent Optimizer
    opt = SGD(
        lr=TRAINING_PARAMETERS['sgd_learning_rate'],
        momentum=TRAINING_PARAMETERS['sgd_momentum']
    )

    # Train Model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model_history = model.fit(x_train, y_train,
                              batch_size=TRAINING_PARAMETERS['batch_size'],
                              epochs=TRAINING_PARAMETERS['epochs'],
                              validation_data=(x_test, y_test),
                              shuffle=True)

    return model, model_history


def save_model(model, model_name):
    """
    Saves trained model as .h5 file
    """

    model_path = os.path.join(CONSTANTS['saved_models_dir'], f"{model_name}.h5")
    model.save(model_path)


def evaluate_model(model, x_test, y_test):
    """
    Evaluates trained model's loss and accuracy on test dataset and prints results
    """

    loss, accuracy = model.evaluate(x_test, y_test)
    print('Model test loss:', loss)
    print('Model test accuracy:', accuracy)
