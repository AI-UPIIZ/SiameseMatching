import os
import pandas as pd
from loguru import logger
from tqdm import tqdm
import numpy as np
from consts.consts import ImageCategories
from consts.paths import Dataset

def generate(dataset_path: str, metadata_file_path: str):
    ''' 
    Generates a CSV file with metadata for images in a specified dataset, including a "split" column.

    Parameters:
        dataset_path : str
            The location of the dataset from which the CSV will be generated (Dataset)
        metadata_file_path : str
            The location where the comma-separated value file (.csv) will be saved (Metadata)
    '''
    logger.info("Generating CSV file with metadata")
    
    data = [
        {
            "person_id": "person_" + filename.split("_")[0],
            "image_name": filename,
            "category": ImageCategories.SELFIE if ImageCategories.SELFIE.value in filename else ImageCategories.ID
        }
        for filename in tqdm(os.listdir(dataset_path), desc="Processing images")
    ]
    
    df_metadata = pd.DataFrame(data)
    
    df_metadata = train_test_split(df_metadata)
    
    df_metadata.sort_values(by='person_id', inplace=True)
    df_metadata.to_csv(metadata_file_path, index=False)
    
    logger.info(f"Metadata saved to {metadata_file_path}")


def train_test_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating train and test split...")
    
    unique_person_ids = dataframe['person_id'].unique()
    np.random.seed(42)  # Ensure reproducibility
    train_ids = np.random.choice(unique_person_ids, size=int(0.8 * len(unique_person_ids)), replace=False)
    dataframe['split'] = dataframe['person_id'].apply(lambda x: 'train' if x in train_ids else 'test')
    
    return dataframe


if __name__ == '__main__':
    generate(Dataset.processed_dataset, Dataset.processed_dataset_metadata)
