from src.predict import load_saved_model, predict_class, load_image


# Main file to run to perform a prediction using a model and an image defined below


model_name = "2_blocks_20_epochs"
image_name = "automobile"

# Load model and image
model = load_saved_model(model_name)
image = load_image(image_name)

# Obtain network predictions
prediction = predict_class(image, model)
print(prediction)
