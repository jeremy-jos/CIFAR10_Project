from src.load_data import load_data, prepare_data
from src.model import classifier_model
from src.train import train_model, save_model, evaluate_model
from src.utils import plot_learning_curves

model_name = "test_model"

# Load and repare dataset
x_train, y_train, x_test, y_test = load_data()
x_train, y_train, x_test, y_test = prepare_data(x_train, y_train, x_test, y_test)

# Train model on data
model = classifier_model(x_train)
trained_model, model_history = train_model(model, x_train, y_train, x_test, y_test)

# Save model and output results
save_model(trained_model, model_name)
evaluate_model(trained_model, x_test, y_test)
plot_learning_curves(model_history, model_name)
