import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_data():
    api = KaggleApi()
    api.authenticate()

    dataset_name = "nicapotato/womens-ecommerce-clothing-reviews"
    download_path = "data/raw"

    os.makedirs(download_path, exist_ok=True)

    print(f"Download dataset '{dataset_name}'...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)

    print(f"DONE: Download completed and unzip in folder: {download_path}")

if __name__ == "__main__":
    download_kaggle_data()