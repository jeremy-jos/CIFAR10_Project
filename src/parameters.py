CONSTANTS = {
    'num_classes': 10,
    'input_shape': (32, 32, 3),
    'saved_models_dir': 'resources/models',
    'outputs_dir': 'outputs',
    'images_dir': 'resources/images'
}

TRAINING_PARAMETERS = {
    'batch_size': 32,
    'epochs': 20,
    'sgd_learning_rate': 0.001,
    'sgd_momentum': 0.9,
}

CIFAR10_LABELS = {
    '0': 'airplane',
    '1': 'automobile',
    '2': 'bird',
    '3': 'cat',
    '4': 'deer',
    '5': 'dog',
    '6': 'frog',
    '7': 'horse',
    '8': 'ship',
    '9': 'truck',

}
