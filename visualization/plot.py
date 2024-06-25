import os
import matplotlib.pyplot as plt


class Plotter:
	def plot_accuracy(self, train_accuracy, val_accuracy, path, model):
		plt.figure(figsize=(10, 7))
		plt.plot(train_accuracy, color='green', label='train accuracy')
		plt.plot(val_accuracy, color='blue', label='validataion accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.title(f'Accuracy plot using {model} architecture')
		plt.legend()
		plt.savefig(path)

	def plot_loss(self, train_loss, val_loss, path, model):
		plt.figure(figsize=(10, 7))
		plt.plot(train_loss, color='orange', label='train loss')
		plt.plot(val_loss, color='red', label='validataion loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title(f'Loss plot using {model} architecture')
		plt.legend()
		plt.savefig(path)
