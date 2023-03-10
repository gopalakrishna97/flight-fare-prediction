from flightfare import utils
from flightfare.entity import config_entity
from flightfare.entity import artifact_entity
from flightfare.exception import FlightFareException
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from flightfare.logger import logging
class DataIngestion():
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise FlightFareException(e)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            df:pd.DataFrame = utils.get_collection_as_datafarme(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name
            )
            # test_df:pd.DataFrame = utils.get_collection_as_datafarme(
            #     database_name=self.data_ingestion_config.database_name,
            #     collection_name=self.data_ingestion_config.test_collection_name
            # )

            logging.info("Save data in feature store")
           
            #replace na with Nan
            df.replace(to_replace="na",value=np.NAN,inplace=True)
            # test_df.replace(to_replace="na",value=np.NAN,inplace=True)


            logging.info("Create feature store folder if not available")

            #Create feature store folder if not available
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)
            
            logging.info("split dataset into train and test set")
            #split dataset into train and test set
            # print(df)
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=30)

            logging.info("create dataset directory folder if not available")
            #create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            train_df = pd.DataFrame(train_df)
            test_df = pd.DataFrame(test_df)

            logging.info(f"train and test columns {train_df.columns}===={test_df.columns}")
            logging.info(f"train and test columns {train_df['Airline'].unique()}===={test_df['Airline'].unique()}")

            pd.DataFrame(train_df).to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            pd.DataFrame(test_df).to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)

            #Prepare artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise FlightFareException(e)