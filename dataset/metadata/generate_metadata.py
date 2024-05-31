import os
import pandas as pd
from loguru import logger
from tqdm import tqdm
import numpy as np
from PIL import Image
from consts.paths import DatasetPaths
from consts.consts import ImageCategories, DatasetSplit

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
    
    data = []
    for filename in tqdm(os.listdir(dataset_path), desc="Processing images"):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Ensure only image files are processed
            file_path = os.path.join(dataset_path, filename)
            with Image.open(file_path) as img:
                width, height = img.size
            data.append({
                "person_id": "person_" + filename.split("_")[0],
                "image_name": filename,
                "category": ImageCategories.SELFIE if "selfie" in filename else ImageCategories.ID,
                "width": width,
                "height": height
            })
    
    df_metadata = pd.DataFrame(data)
    
    # Check for duplicates and keep only the first occurrence
    df_metadata = df_metadata.drop_duplicates(subset=['person_id', 'category'], keep='first')
    
    # Pivot the table to have selfie and idcard in separate columns
    df_metadata_pivot = df_metadata.pivot(index='person_id', columns='category', values=['image_name', 'width', 'height']).reset_index()
    
    # Flatten the columns after pivoting
    df_metadata_pivot.columns = ['person_id', 'selfie_image_name', 'idcard_image_name', 'selfie_width', 'idcard_width', 'selfie_height', 'idcard_height']
    
    # Generate the train/test split
    df_metadata_split = train_test_split(df_metadata_pivot)
    
    df_metadata_split.sort_values(by='person_id', inplace=True)
    df_metadata_split.to_csv(metadata_file_path, index=False)
    
    logger.info(f"Metadata saved to {metadata_file_path}")

def train_test_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating train and test split...")
    
    unique_person_ids = dataframe['person_id'].unique()
    np.random.seed(42)  # Ensure reproducibility
    train_ids = np.random.choice(unique_person_ids, size=int(0.8 * len(unique_person_ids)), replace=False)
    dataframe['split'] = dataframe['person_id'].apply(lambda x: DatasetSplit.TRAIN if x in train_ids else DatasetSplit.TEST)
    
    return dataframe

if __name__ == '__main__':
    generate(DatasetPaths.processed_dataset, DatasetPaths.processed_dataset_metadata)
