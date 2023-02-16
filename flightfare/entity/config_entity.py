import os
from datetime import datetime
from flightfare.exception import FlightFareException
FILE_NAME = "flightfare.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
MODEL_FILE_NAME = "model.pkl"

class TrainingPipeLineConfig:

    def __init__(self):
        self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y_%H%M%S')}")


class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipeLineConfig):
        self.database_name = "flightfare"
        self.collection_name = "flightdata"
        self.test_collection_name = 'flighttestdata'
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir,"data_ingestion")
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
        self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
        self.test_size = 0.2
    def to_dict(self)->dict:
        try:
            return self.__dict__
        except Exception as e:
            raise FlightFareException(e)


class DatavalidationConfig:

    def __init__(self,training_pipeline_config:TrainingPipeLineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_validation")
        self.report_file_path = os.path.join(self.data_validation_dir,"report.yaml")
        self.missing_threshold:float = 0.2
        self.base_file_path = os.path.join("Data_Train.xlsx")


class DataTransformationConfig:
   def __init__(self,training_pipeline_config:TrainingPipeLineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_transformation")
        self.transform_object_path = os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path =  os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME.replace("csv","npz"))
        self.transformed_test_path =os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace("csv","npz"))


class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipeLineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir , "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
        self.expected_score = 0.7
        self.overfitting_threshold = 0.16


class ModelEvaluationConfig:
    def __init__(self,training_pipeline_config:TrainingPipeLineConfig):
        self.change_threshold = 0.01

        
class ModelPusherConfig:
    def __init__(self,training_pipeline_config:TrainingPipeLineConfig):

        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir , "model_pusher")
        self.saved_model_dir = os.path.join('saved_models')
        self.pusher_model_dir = os.path.join(self.model_pusher_dir,"saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir,TRANSFORMER_OBJECT_FILE_NAME)













