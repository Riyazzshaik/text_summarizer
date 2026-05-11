from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_evalution import ModelEvaluation
from textSummarizer.logging import logger


class ModelEvalutionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config= ConfigurationManager()
        model_evalution_config = config.get_model_evaluation_config()
        model_evalution_config =ModelEvaluation(config=model_evalution_config)
        model_evalution_config.evaluate()