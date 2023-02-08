from flightfare.logger import logging
from flightfare.exception import FlightFareException
from flightfare.utils import get_collection_as_datafarme
import sys,os
from flightfare.entity.config_entity import DataIngestionConfig
from flightfare.entity import config_entity
from flightfare.components.data_ingestion import DataIngestion
from flightfare.components.data_validation import DataValidation
from flightfare.components.data_transformation import DataTransformation
from flightfare.components.model_trainer import ModelTrainer
def test_logger_exception():
    try:
        # data ingestion
        training_pipeline_config = config_entity.TrainingPipeLineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # data validation
        data_validation_config = config_entity.DatavalidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                                         data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()

        # data transformation
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_tranformationconfig=data_transformation_config,
                                         data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        # model trainer
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                    data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
    except Exception as e:
        logging.error(str(e))
        raise FlightFareException(e)

if __name__=="__main__":
    try:
        test_logger_exception()
    except Exception as e:
        print(e)    