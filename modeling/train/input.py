from torch.utils.data import DataLoader
from consts.consts import DatasetSplit


class InputPipeline:
	def __init__(self, datasets_list, batch_size, pin_memory):
		self.datasets = {
			DatasetSplit.TRAIN: datasets_list[0],
			DatasetSplit.VAL: datasets_list[1],
		}
		self.batch_size = batch_size
		self.pin_memory = pin_memory

		self.data_loaders = {
			DatasetSplit.TRAIN: DataLoader(
				self.datasets[DatasetSplit.TRAIN],
				batch_size=self.batch_size,
				shuffle=True,
				pin_memory=self.pin_memory,
			),
			DatasetSplit.VAL: DataLoader(
				self.datasets[DatasetSplit.VAL],
				batch_size=self.batch_size,
				shuffle=False,
				pin_memory=self.pin_memory,
			),
		}

	def __getitem__(self, split):
		return self.data_loaders.get(split, None)

	def __contains__(self, split):
		return split in self.data_loaders
