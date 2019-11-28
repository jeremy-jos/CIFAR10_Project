import matplotlib.pyplot as plt

from src.parameters import CONSTANTS


def plot_learning_curves(model_history, model_name):

	# plot loss learning curve
	plt.subplot(211)
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.plot(model_history.history['loss'], color='blue', label='train')
	plt.plot(model_history.history['val_loss'], color='orange', label='test')

	# plot accuracy learning curve
	plt.subplot(212)
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.plot(model_history.history['accuracy'], color='blue', label='train')
	plt.plot(model_history.history['val_accuracy'], color='orange', label='test')

	# save plot to file
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
	plots_file = f"{CONSTANTS['outputs_dir']}/{model_name}_plots.png"
	plt.savefig(plots_file)
	plt.close()
