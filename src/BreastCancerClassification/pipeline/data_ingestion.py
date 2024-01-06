from src.BreastCancerClassification.config.configuration import ConfigurationManager
from src.BreastCancerClassification.components.data_ingestion import DataIngestion


STAGE_NAME = "Data Ingestion stage"


class DataIngestionPipeline:
    def __init__(self):
        config = ConfigurationManager().get_data_ingestion_config()
        data_ingestion = DataIngestion(config=config)
        data_ingestion.extract_tarfile()

