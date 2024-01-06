import os
from pathlib import Path
import fnmatch
import random
from BreastCancerClassification.entity.config_entity import DataSplitConfig
from BreastCancerClassification import logger
from BreastCancerClassification.utils.common import create_directories, copy_file


class DataSplitting:
    def __init__(self, split_config: DataSplitConfig):
        self.split_config = split_config

        self.benign_path = None
        self.malignant_path = None
        for root, dirs, files in os.walk(self.split_config.source_data_dir):
            if "benign" in dirs:
                self.benign_path = os.path.join(root, "benign")
            if "malignant" in dirs:
                self.malignant_path = os.path.join(root, "malignant")
        if self.benign_path is None:
            logger.info("Benign folder Not Found.")
        if self.malignant_path is None:
            logger.info("Malignant folder Not Found.")

    def find_png_files(self, directory: Path, zoom_factor: str):
        if zoom_factor not in ['all', '40x', '100x', '200x', '400x']:
            logger.info("Invalid zoom factor. Reset to 'all'.")
            zoom_factor = 'all'
        png_files = []
        for root, dirs, files in os.walk(directory):
            for file in fnmatch.filter(files, '*.png'):
                full_path = os.path.join(root, file)
                if zoom_factor in full_path:
                    png_files.append(full_path)
                png_files.append(full_path)
        return png_files

    def do_split(self, data_list: list):
        random.shuffle(data_list)
        total = len(data_list)
        tr, tst, vl = [int((i / 100) * total) for i in self.split_config.param_train_test_val_ratio]
        try:
            train_set = data_list[0:tr]
            test_set = data_list[tr:tr + tst]
            val_set = data_list[tr + tst:total]

            if len(train_set) == 0 or len(test_set) == 0 or len(val_set) == 0:
                logger.info("Invalid Splitting.")
                return None, None, None
        except Exception as e:
            logger.exception(e)
            return None, None, None
        return train_set, test_set, val_set

    def copy_data(self, source_list, dest_dir, dest_subfolder, file_prefix):
        for i, source_path in enumerate(source_list):
            source_path = Path(source_path)
            destination_dir = dest_dir / dest_subfolder
            create_directories([destination_dir], False)
            file_name = f"{file_prefix}{i}.png"
            copy_file(source_path, destination_dir, file_name)

    def train_test_val_split(self):
        ben_list = self.find_png_files(Path(self.benign_path), self.split_config.param_zoom_factor)
        mal_list = self.find_png_files(Path(self.malignant_path), self.split_config.param_zoom_factor)
        train_ben, test_ben, val_ben = self.do_split(ben_list)
        train_mal, test_mal, val_mal = self.do_split(mal_list)
        if train_ben is None or train_mal is None:
            return False

        create_directories(
            [self.split_config.config_train_dir, self.split_config.config_test_dir, self.split_config.config_val_dir])
        self.copy_data(train_ben, self.split_config.config_train_dir, "ben", "Train_Benign")
        self.copy_data(train_mal, self.split_config.config_train_dir, "mal", "Train_Malignant")
        self.copy_data(test_ben, self.split_config.config_test_dir, "ben", "Test_Benign")
        self.copy_data(test_mal, self.split_config.config_test_dir, "mal", "Test_Malignant")
        self.copy_data(val_ben, self.split_config.config_val_dir, "ben", "Val_Benign")
        self.copy_data(val_mal, self.split_config.config_val_dir, "mal", "Val_Malignant")
        return True
