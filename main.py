from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.logging import logger
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

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