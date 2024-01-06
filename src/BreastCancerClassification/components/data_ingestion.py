import os
import tarfile
from pathlib import Path
from BreastCancerClassification.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def extract_tarfile(self):
        extraction_path = Path(self.config.config_extract_dir)
        compressed_file_path = Path(self.config.config_compressed_file)
        os.makedirs(extraction_path, exist_ok=True)
        with tarfile.open(compressed_file_path) as tar:
            tar.extractall(extraction_path)


