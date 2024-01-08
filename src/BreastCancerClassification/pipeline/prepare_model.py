from BreastCancerClassification.config.configuration import ConfigurationManager
from BreastCancerClassification.components.prepare_model import PrepareModel
from BreastCancerClassification import logger

STAGE_NAME = "Prepare Model"


class PrepareModelPipeline:
    def __init__(self):
        configs = ConfigurationManager()
        model_config = configs.get_model_config()
        prepare_model = PrepareModel(config=model_config)
        prepare_model.build_model()
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_model = PrepareModelPipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e