import os

from keras.optimizers import SGD

from src.parameters import CONSTANTS, TRAINING_PARAMETERS


def train_model(model, x_train, y_train, x_test, y_test):

    # initiate RMSprop optimizer
    opt = SGD(lr=0.001, momentum=0.9)

    # Let's train the model using RMSprop
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

    model_path = os.path.join(CONSTANTS['saved_models_dir'], model_name)
    model.save(model_path)


def evaluate_model(model, x_test, y_test):

    loss, accuracy = model.evaluate(x_test, y_test)
    print('Model test loss:', loss)
    print('Model test accuracy:', accuracy)
