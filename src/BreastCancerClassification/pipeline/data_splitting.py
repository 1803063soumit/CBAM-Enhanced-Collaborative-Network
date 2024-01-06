from BreastCancerClassification.config.configuration import ConfigurationManager
from BreastCancerClassification.components.data_splitting import DataSplitting


STAGE_NAME = "Data Splitting stage"


class DataSplittingPipeline:
    def __init__(self):
        configs = ConfigurationManager()
        data_splitting_config = configs.get_data_split_config()
        data_splitting = DataSplitting(split_config=data_splitting_config)
        data_splitting.train_test_val_split()
