from BreastCancerClassification.config.configuration import ConfigurationManager
from BreastCancerClassification.components.evaluate_model import Evaluation




STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        # evaluation.log_into_mlflow()
