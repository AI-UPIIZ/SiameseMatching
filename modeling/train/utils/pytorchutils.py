import os
import subprocess

from loguru import logger


class EarlyStopping:
	def __init__(self, patience=5, min_delta=0):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, val_loss):
		if self.best_loss is None:
			self.best_loss = val_loss
		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			logger.info(f'Early stopping counter {self.counter} of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True


def squeeze_generic(a, axes_to_keep):
	out_s = [s for i, s in enumerate(a.shape) if i in axes_to_keep or s != 1]
	return a.reshape(out_s)


def current_memory_usage():
	out = (
		subprocess.Popen(
			['ps', '-p', str(os.getpid()), '-o', 'rss'], stdout=subprocess.PIPE
		)
		.communicate()[0]
		.split(b'\n')
	)
	mem = float(out[1].strip()) / 1024
	return mem
