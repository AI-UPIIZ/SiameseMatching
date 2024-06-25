import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from consts.consts import DatasetSplit

from loguru import logger


class BasePreprocess(Dataset):
	def __init__(
		self, subset, image_paths, image_labels, target_class=False, filename=False
	):
		assert subset in [
			attr for attr in dir(DatasetSplit) if not attr.startswith('__')
		]
		if target_class:
			assert isinstance(target_class, list)
		if filename:
			assert isinstance(filename, list)
		self.subset = subset
		self.image_paths = image_paths
		self.image_labels = image_labels
		self.target_class = target_class
		self.filename = filename
		self.transform = transforms.Compose(self._get_transforms_list())
		self.label_transformation = None
		self.n_outputs = len(set(image_labels))
		self.set_up_label_transformation_for_classification()

	def set_up_label_transformation_for_classification(self):
		sorted_labels = sorted(set(self.image_labels))
		label2idx = {raw_label: idx for idx, raw_label in enumerate(sorted_labels)}
		self.label_transformation = lambda x: torch.tensor(
			label2idx[x], dtype=torch.long
		).view(-1, 1)

	def _get_transforms_list(self):
		return [
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]

	def get_n_outputs(self):
		return self.n_outputs

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		img_path = self.image_paths[idx]
		image = Image.open(img_path).convert('RGB')
		label = self.image_labels[idx]
		if self.label_transformation:
			label = self.label_transformation(label)
		image = self.transform(image)
		_return = [image, label]
		if self.target_class:
			_return.append(self.target_class[idx])
		if self.filename:
			_return.append(self.filename[idx])
		return _return


class SiamesePreprocess(Dataset):
	def __init__(self, metadata_csv, split, processed_dataset):
		self.metadata = pd.read_csv(metadata_csv)
		self.metadata = self.metadata[self.metadata['subset'] == split]
		self.transform = transforms.Compose(
			[
				transforms.Resize((128, 128)),
				transforms.ToTensor(),
			]
		)
		self.split = split
		self.processed_dataset = processed_dataset

	def load_image(self, image_path):
		if not isinstance(image_path, str):
			raise ValueError(f"Invalid image path: {image_path}")

		full_path = os.path.join(self.processed_dataset, image_path)
		if not os.path.exists(full_path):
			raise FileNotFoundError(f"Image file not found: {full_path}")
		
		image = Image.open(full_path).convert('RGB')
		return self.transform(image)

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, index):
		row = self.metadata.iloc[index]
		selfie_path = row['selfie_img_name']
		idcard_path = row['idcard_img_name']

		selfie_image = self.load_image(selfie_path)
		idcard_image = self.load_image(idcard_path)
		label = torch.tensor(row['label'], dtype=torch.float32)

		return selfie_image, idcard_image, label

	def get_n_outputs(self):
		return self.metadata['label'].nunique()
