from src.load_data import load_data, prepare_data
from src.model import classifier_model
from src.predict import load_saved_model, predict_class, load_image
from src.train import train_model, save_model, evaluate_model
from src.utils import plot_learning_curves


def train_main(model_name):

    # Load and prepare dataset
    x_train, y_train, x_test, y_test = load_data()
    x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test)

    # Train model on data
    model = classifier_model()
    trained_model, model_history = train_model(model, x_train, y_train, x_test, y_test)

    # Save model and output results
    print("ok")
    save_model(trained_model, model_name)
    evaluate_model(trained_model, x_test, y_test)
    plot_learning_curves(model_history, model_name)


def predict_main(image_name, model_name):

    # Load model and image
    model = load_saved_model(model_name)
    image = load_image(image_name)

    # Obtain network predictions
    predictions = predict_class(image, model)

    # Process network prediction
    print(predictions[0])


train_main("2_blocks_20_epochs")

#predict_main("car", "test_model_new")
