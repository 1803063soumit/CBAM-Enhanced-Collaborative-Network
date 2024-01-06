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
