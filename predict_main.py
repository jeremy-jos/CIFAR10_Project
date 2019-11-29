from src.predict import load_saved_model, predict_class, load_image


# Main file to run to perform a prediction using a model and an image defined below


model_name = "test_model_new"
image_name = "car"

# Load model and image
model = load_saved_model(model_name)
image = load_image(image_name)

# Obtain network predictions
predictions = predict_class(image, model)

# Process network prediction
print(predictions[0])
