from BreastCancerClassification import logger
from BreastCancerClassification.pipeline.data_ingestion import DataIngestionPipeline
from BreastCancerClassification.pipeline.data_splitting import DataSplittingPipeline
from BreastCancerClassification.pipeline.prepare_model import PrepareModelPipeline
from BreastCancerClassification.pipeline.training_model import ModelTrainingPipeline
# STAGE_NAME = "Data Ingestion stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    data_ingestion=DataIngestionPipeline()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

# STAGE_NAME = "Data Splitting stage"
# try:
#    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#    data_splitting=DataSplittingPipeline()
#    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#         logger.exception(e)
#         raise e

STAGE_NAME = "Prepare Model"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_model = PrepareModelPipeline()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Training"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
