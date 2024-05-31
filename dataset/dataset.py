import os
from typing import Tuple
import pandas as pd

from consts.paths import DatasetPaths
from consts.consts import DatasetSplit


class Dataset:
    def __init__(self, csv_path: str, data_folder: str) -> None:
        self.data_folder = data_folder
        self.df = pd.read_csv(csv_path)

    def get_subset(self, subset: DatasetSplit) -> Tuple:
        assert subset in DatasetSplit
        df_subset = self.df[self.df["subset"] == subset.value]
        image1_name = df_subset["selfie"].tolist()
        image2_name = df_subset["idcard"].tolist()
        image_labels = df_subset["person_id"].tolist()
        image1_filepaths = self.get_filepaths(image1_name)
        image2_filepaths = self.get_filepaths(image2_name)
        return [image1_filepaths, image2_filepaths], image_labels

    def get_n_outputs(self) -> int:
        _, image_labels = self.get_subset(DatasetSplit.TRAIN)
        unique_labels = len(set(image_labels))
        return unique_labels

    def get_filepaths(self, image_filenames: list) -> list:
        return [
            os.path.join(self.data_folder, image_filename)
            for image_filename in image_filenames
        ]

    @classmethod
    def load_dataset(cls):
        raise NotImplementedError


class PollinatorsDataset(Dataset):
    @classmethod
    def load_dataset(cls):
        csv_path = DatasetPaths.processed_dataset_metadata
        data_folder = DatasetPaths.processed_dataset
        return cls(csv_path, data_folder)
