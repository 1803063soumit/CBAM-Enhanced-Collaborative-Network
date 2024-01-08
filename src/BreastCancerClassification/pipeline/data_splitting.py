from BreastCancerClassification.config.configuration import ConfigurationManager
from BreastCancerClassification.components.data_splitting import DataSplitting
from BreastCancerClassification import logger

STAGE_NAME = "Data Splitting stage"


class DataSplittingPipeline:
    def __init__(self):
        configs = ConfigurationManager()
        data_splitting_config = configs.get_data_split_config()
        data_splitting = DataSplitting(split_config=data_splitting_config)
        data_splitting.train_test_val_split()

if __name__ == '__main__':
    try:
       logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
       data_splitting=DataSplittingPipeline()
       logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            logger.exception(e)
            raise e