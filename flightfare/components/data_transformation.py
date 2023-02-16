from flightfare.entity import artifact_entity,config_entity
from flightfare.exception import FlightFareException
from flightfare.logger import logging
import os, sys
from typing import Optional
import pandas as pd
import numpy as np
from flightfare import utils
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from flightfare.db_config import TARGET_COLUMN


class DataTransformation:

    def __init__(self,data_tranformationconfig:config_entity.DataTransformationConfig,
                        data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            self.data_tranformationconfig = data_tranformationconfig
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise FlightFareException(e)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler =  RobustScaler()
            pipeline = Pipeline(steps=[
                    ('Imputer',simple_imputer),
                    ('RobustScaler',robust_scaler)
                ])
            return pipeline
        except Exception as e:
            raise FlightFareException(e)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = utils.extracting_new_columns(train_df,df_type="train")
            test_df = utils.extracting_new_columns(test_df,df_type="test")
            logging.info(f"{train_df.columns} and {test_df.columns}")
            #selecting input feature for train and test dataframe
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            #selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            #transforming input features
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)

            target_feature_train_arr = target_feature_train_df.values
            target_feature_test_arr = target_feature_test_df.values

            

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_tranformationconfig.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_tranformationconfig.transformed_test_path,
                                        array=test_arr)

            utils.save_object(file_path=self.data_tranformationconfig.transform_object_path,
             obj=transformation_pipleine)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_tranformationconfig.transform_object_path,
                transformed_train_path = self.data_tranformationconfig.transformed_train_path,
                transformed_test_path = self.data_tranformationconfig.transformed_test_path,
                )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise FlightFareException(e)