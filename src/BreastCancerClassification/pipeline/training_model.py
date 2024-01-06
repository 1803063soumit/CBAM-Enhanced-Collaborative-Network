from BreastCancerClassification.components.training_model import Training
from BreastCancerClassification.config.configuration import ConfigurationManager

STAGE_NAME = "Training Model"


class ModelTrainingPipeline:
    def __init__(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        model_config = config.get_model_config()
        data_split_config = config.get_data_split_config()

        training = Training(training_config=training_config, model_config=model_config,
                            data_split_config=data_split_config)
        training.train()
