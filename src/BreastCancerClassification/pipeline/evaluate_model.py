from BreastCancerClassification.config.configuration import ConfigurationManager
from BreastCancerClassification.components.evaluate_model import Evaluation
from BreastCancerClassification import  logger
STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evalution = EvaluationPipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e