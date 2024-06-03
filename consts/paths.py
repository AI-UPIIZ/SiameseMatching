import os

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir,) * 2))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
RAW_DATA = os.path.join(DATA_PATH, 'raw')
INTER_DATA = os.path.join(DATA_PATH, 'inter')
PROCESSED_DATA = os.path.join(DATA_PATH, 'processed')
CONFIG_PATH = os.path.join(PROJECT_PATH, 'config')
ARTIFACTS_PATH = os.path.join(PROJECT_PATH, 'artifacts')


class DatasetPaths:
	raw_dataset = os.path.join(RAW_DATA, 'dataset')
	processed_dataset = os.path.join(PROCESSED_DATA, 'dataset')
	processed_dataset_metadata = os.path.join(processed_dataset, 'metadata.csv')


class ModelPaths:
	models_path = os.path.join(ARTIFACTS_PATH, 'models')
