from src.load_data import load_data, prepare_data
from src.model import classifier_model
from src.train import train_model, save_model, evaluate_model
from src.utils import plot_learning_curves

model_name = "test_model"

x_train, y_train, x_test, y_test = load_data()
x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test)

model = classifier_model(x_train)
trained_model, model_history = train_model(model, x_train, y_train, x_test, y_test)

save_model(trained_model, f"{model_name}.h5")
evaluate_model(trained_model, x_test, y_test)
plot_learning_curves(model_history, model_name)
