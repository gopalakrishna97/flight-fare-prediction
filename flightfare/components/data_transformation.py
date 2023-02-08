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


    def extracting_new_columns(self,df:pd.DataFrame,df_type:str)->pd.DataFrame:
        """
        This method accepts pandas dataframe and df_type like train or test,
        Then extract the new coulns out of existing columns and deletes the old columns

        Returns DataFrame
        """
        # Preprocessing

        logging.info(f"{df_type} data Info")
        print("-"*75)
        print(df.info())

        logging.info(f"Null values in {df_type} :")

        df.dropna(inplace = True)
        logging.info(f"sum of null values in {df_type}: {df.isnull().sum()}")

        # EDA

        # Date_of_Journey
        logging.info(f"Extracting Journey_day, Journey_month from Date_of_Journey column")
        df["Journey_day"] = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.day
        df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format = "%d/%m/%Y").dt.month
        df.drop(["Date_of_Journey"], axis = 1, inplace = True)

        # Dep_Time
        logging.info(f"Extracting Dep_hour, Dep_min from Dep_Time")
        df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
        df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
        df.drop(["Dep_Time"], axis = 1, inplace = True)

        # Arrival_Time
        logging.info(f"Extracting Arrival_hour, Arrival_min from Arrival_Time column")
        df["Arrival_hour"] = pd.to_datetime(df.Arrival_Time).dt.hour
        df["Arrival_min"] = pd.to_datetime(df.Arrival_Time).dt.minute
        df.drop(["Arrival_Time"], axis = 1, inplace = True)

        # Duration
        logging.info(f"Extracting the duration_hours,duration_mins from Duration column")
        duration = list(df["Duration"])

        for i in range(len(duration)):
            if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
                if "h" in duration[i]:
                    duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
                else:
                    duration[i] = "0h " + duration[i]           # Adds 0 hour

        duration_hours = []
        duration_mins = []
        for i in range(len(duration)):
            duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
            duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

        # Adding Duration column to test set
        df["Duration_hours"] = duration_hours
        df["Duration_mins"] = duration_mins
        df.drop(["Duration"], axis = 1, inplace = True)


        # Categorical data

        logging.info("Airline")
        logging.info("-"*75)
        logging.info(df["Airline"].value_counts())
        Airline = pd.get_dummies(df["Airline"], drop_first= True)


        logging.info("Source")
        logging.info("-"*75)
        logging.info(df["Source"].value_counts())
        Source = pd.get_dummies(df["Source"], drop_first= True)

        logging.info("Destination")
        logging.info("-"*75)
        logging.info(df["Destination"].value_counts())
        Destination = pd.get_dummies(df["Destination"], drop_first = True)

        # Additional_Info contains almost 80% no_info
        # Route and Total_Stops are related to each other
        df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

        # Replacing Total_Stops
        df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

        # Concatenate dataframe --> df + Airline + Source + Destination
        df = pd.concat([df, Airline, Source, Destination], axis = 1)

        df.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)



        logging.info(f"Shape of {df_type} data :  {df.shape}")
        logging.info("-"*75)
        return df

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.extracting_new_columns(train_df,df_type="train")
            test_df = self.extracting_new_columns(test_df,df_type="test")
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