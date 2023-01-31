from flightfare.logger import logging
from flightfare.exception import FlightFareException
from flightfare.utils import get_collection_as_datafarme
import sys,os
from flightfare.entity.config_entity import DataIngestionConfig
from flightfare.entity import config_entity
from flightfare.components.data_ingestion import DataIngestion

def test_logger_exception():
    try:
        # print("test")
        # get_collection_as_datafarme(database_name="flightfare",collection_name="flightdata")
        training_pipeline_config = config_entity.TrainingPipeLineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        # print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        print(data_ingestion.initiate_data_ingestion())
    except Exception as e:
        logging.error(str(e))
        raise FlightFareException(e)

if __name__=="__main__":
    try:
        test_logger_exception()
    except Exception as e:
        print(e)    