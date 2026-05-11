from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.logging import logger
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipeline.stage_05_model_evaluation import ModelEvalutionTrainingPipeline
STAGE_NAME =  "Data  Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started  <<<<<")
    data_ingestion =DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME}  completed  <<<<<< \n\n X=========X")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME =  "Data  validation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started  <<<<<")
    data_validation =DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>> stage {STAGE_NAME}  completed  <<<<<< \n\n X=========X")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME =  "Data  transformation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started  <<<<<")
    data_transformation =DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>> stage {STAGE_NAME}  completed  <<<<<< \n\n X=========X")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME =  "model trainer Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started  <<<<<")
    model_trainer =ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>> stage {STAGE_NAME}  completed  <<<<<< \n\n X=========X")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME =  "model evalutionr Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started  <<<<<")
    model_evalution =ModelEvalutionTrainingPipeline()
    model_evalution.main()
    logger.info(f">>>>> stage {STAGE_NAME}  completed  <<<<<< \n\n X=========X")

except Exception as e:
    logger.exception(e)
    raise e