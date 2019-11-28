import cv2
import numpy as np
from keras.models import load_model

from src.parameters import CONSTANTS


def load_saved_model(model_name):
    """
    Load model saved as .h5 file
    """
    model_path = f"{CONSTANTS['saved_models_dir']}/{model_name}.h5"
    loaded_model = load_model(model_path)

    return loaded_model


def load_image(image_name):
    """
    Load and resize and image to match the input shape of the model
    """

    image_path = f"{CONSTANTS['images_dir']}/{image_name}.jpg"
    image = cv2.imread(image_path)
    return cv2.resize(image, dsize=CONSTANTS['input_shape'][:2], interpolation=cv2.INTER_CUBIC)


def predict_class(image, model):
    """
    Predict an image class with a model
    """

    # Convert image to right input format
    image_input = np.array([image])
    return model.predict(image_input)
