from BreastCancerClassification.config.configuration import ConfigurationManager
from BreastCancerClassification.components.data_ingestion import DataIngestion
from BreastCancerClassification import logger

STAGE_NAME = "Data Ingestion stage"


class DataIngestionPipeline:
    def __init__(self):
        config = ConfigurationManager().get_data_ingestion_config()
        data_ingestion = DataIngestion(config=config)
        data_ingestion.extract_tarfile()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion = DataIngestionPipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
