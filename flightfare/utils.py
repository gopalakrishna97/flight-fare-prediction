import pandas as pd
from flightfare.logger import logging
from flightfare.exception import FlightFareException
from flightfare.db_config import mongo_client
import os
import yaml
import numpy as np
import dill
from datetime import datetime

def get_collection_as_datafarme(database_name:str,collection_name:str)->pd.DataFrame:
    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id ")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and columns in df: {df.shape}")
        return df
    except Exception as e:
        raise FlightFareException(e)



def write_yaml_file(file_path:str,data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(file_path,'w') as file:
            yaml.dump(data,file)
    except Exception as e:
        raise FlightFareException(e)

def convert_columns_float(df,exclude_columns:list)->pd.DataFrame:
    try:
        for column in df.columns:
            if column not in exclude_columns:
                if column == "Route" or "Total_Stops":
                    df[column]=df[column].astype(str)
        return df
    except Exception as e:
        raise e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise FlightFareException(e) from e


def load_object(file_path:str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise FlightFareException(e) from e


def save_numpy_array_data(file_path: str, array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise FlightFareException(e) from e

def load_numpy_array_data(file_path: str):
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise FlightFareException(e) from e


def extracting_new_columns(df:pd.DataFrame,df_type:str)->pd.DataFrame:
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