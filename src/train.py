import os

from keras.optimizers import RMSprop

from src.parameters import TRAINING_PARAMETERS


def train_model(model, x_train, y_train, x_test, y_test):

    # initiate RMSprop optimizer
    opt = RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=TRAINING_PARAMETERS['batch_size'],
              epochs=TRAINING_PARAMETERS['epochs'],
              validation_data=(x_test, y_test),
              shuffle=True)

    return model


def save_model(model, model_name):

    model_path = os.path.join(TRAINING_PARAMETERS['saved_models_dir'], model_name)
    model.save(model_path)


def evaluate_model(model, x_test, y_test):

    loss, accuracy = model.evaluate(x_test, y_test)
    print('Model test accuracy:', accuracy)
