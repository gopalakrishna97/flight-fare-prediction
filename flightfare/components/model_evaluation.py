from flightfare.entity import artifact_entity,config_entity
from flightfare.exception import FlightFareException
from flightfare.predictor import ModelResolver
from flightfare.logger import logging
import os, sys
from typing import Optional
import pandas as pd
import numpy as np
from flightfare.utils import load_object
from flightfare import utils
from sklearn.metrics import r2_score
from flightfare.db_config import TARGET_COLUMN


class ModelEvaluation:

    def __init__(self,
        model_eval_config:config_entity.ModelEvaluationConfig,
        data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
        data_transformation_artifact:artifact_entity.DataTransformationArtifact,
        model_trainer_artifact:artifact_entity.ModelTrainerArtifact      
        ):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise FlightFareException(e)
    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        # if we have model in saved_models folder then we will compare the current_trained model and model in saved_model folder 
        try:
            logging.info("if we have model in saved_models folder then we will compare the\
                            current_trained model and model in saved_model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_eval_artifact  = artifact_entity.ModelEvaluationArtifact(is_model_accepted = True,improved_accuracy=False)
                logging.info(f"model evaluation artifact is {model_eval_artifact}")
                return model_eval_artifact


            # finding the locations of transformer and model
            logging.info("finding the locations of transformer and model")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()


            #  load the model using above file lovcations
            # these are previously trained models
            logging.info("load the model using above file lovcations these are previously trained models")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)


            # currently trained models
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)


            # test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            # target_df = test_df[TARGET_COLUMN]

            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]
            y_true = y_test
            logging.info(f"y_true value from tranformer object {y_true[:5]} ")

            # Accuracy using previously trained models
            # input_arr = transformer.transform(x_test)
            prev_y_pred = model.predict(x_test)
            logging.info(f"y_pred values from model prediction {prev_y_pred[:5]}")
            previous_model_score = r2_score(y_true,prev_y_pred)
            logging.info(f"Accuracy using previously trained models {previous_model_score}")

            # Accuracy using current_model
            # input_arr = current_transformer.transform(x_test)
            curr_y_pred = current_model.predict(x_test)
            logging.info(f"Prediction using current trained model: {curr_y_pred[:5]}")
            current_model_score = r2_score(y_true=y_true,y_pred=curr_y_pred)
            logging.info(f'Accuracy using currently trained model {current_model_score}')


            if current_model_score<=previous_model_score:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=False,fall_accuracy= float(previous_model_score-current_model_score))
                logging.info(f"current trained model is not better than previous model  : {model_eval_artifact}")
                return model_eval_artifact


            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,improved_accuracy= float(current_model_score-previous_model_score))
            logging.info(f"Model evaluation artifact {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise FlightFareException(e)


