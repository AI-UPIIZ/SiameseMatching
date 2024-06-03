import os
import pandas as pd
from loguru import logger
from tqdm import tqdm
import numpy as np
from PIL import Image
from consts.paths import DatasetPaths
from consts.consts import ImageCategories, DatasetSplit, Metadata, ImageFormats


def generate(dataset_path: str, metadata_file_path: str):
	"""
	Generates a CSV file with metadata for images in a specified dataset, including a "split" column.

	Parameters:
	    dataset_path : str
	        The location of the dataset from which the CSV will be generated (Dataset)
	    metadata_file_path : str
	        The location where the comma-separated value file (.csv) will be saved (Metadata)
	"""
	logger.info('Generating CSV file with metadata...')
	data = []
	for filename in tqdm(os.listdir(dataset_path), desc='Processing images'):
		if filename.endswith(
			(ImageFormats.PNG.value, ImageFormats.JPG.value, ImageFormats.JPEG.value)
		):
			file_path = os.path.join(dataset_path, filename)
			with Image.open(file_path) as img:
				width, height = img.size
			data.append(
				{
					Metadata.PERSONID: 'person_' + filename.split('_')[0],
					Metadata.IMG_NAME: filename,
					Metadata.CATEGORY: ImageCategories.SELFIE
					if 'selfie' in filename
					else ImageCategories.ID,
					Metadata.WIDTH: width,
					Metadata.HEIGHT: height,
				}
			)

	df_metadata = pd.DataFrame(data)
	df_metadata = df_metadata.drop_duplicates(
		subset=[Metadata.PERSONID, Metadata.CATEGORY], keep='first'
	)
	df_metadata_pivot = df_metadata.pivot(
		index=Metadata.PERSONID,
		columns=Metadata.CATEGORY,
		values=[Metadata.IMG_NAME, Metadata.WIDTH, Metadata.HEIGHT],
	).reset_index()
	df_metadata_pivot.columns = [
		Metadata.PERSONID,
		Metadata.SELFIE_IMG_NAME,
		Metadata.IDCARD_IMG_NAME,
		Metadata.SELFIE_WIDTH,
		Metadata.IDCARD_WIDTH,
		Metadata.SELFIE_HEIGHT,
		Metadata.IDCARD_HEIGHT,
	]

	df_metadata_split = train_test_split(df_metadata_pivot)

	df_metadata_split.sort_values(by=Metadata.PERSONID, inplace=True)
	df_metadata_split.to_csv(metadata_file_path, index=False)

	logger.info(f'Metadata saved to {metadata_file_path}')


def train_test_split(dataframe: pd.DataFrame) -> pd.DataFrame:
	logger.info('Generating train and test split...')

	unique_person_ids = dataframe[Metadata.PERSONID].unique()
	np.random.seed(42)  # Ensure reproducibility
	train_ids = np.random.choice(
		unique_person_ids, size=int(0.8 * len(unique_person_ids)), replace=False
	)
	dataframe[Metadata.SPLIT] = dataframe[Metadata.PERSONID].apply(
		lambda x: DatasetSplit.TRAIN if x in train_ids else DatasetSplit.TEST
	)

	return dataframe


if __name__ == '__main__':
	generate(DatasetPaths.processed_dataset, DatasetPaths.processed_dataset_metadata)
