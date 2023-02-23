from flightfare.exception import FlightFareException
from flightfare.logger import logging
from flightfare.predictor import ModelResolver
from datetime import datetime
import os
import numpy as np
import pandas as pd 
from flightfare import utils
from typing import Optional
from flightfare.components.data_transformation import DataTransformation
PREDICTION_DIR="prediction"
DATABASE_NAME="flightfare"
TEST_COLLECTION_NAME = "flighttestdata"

def batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        # test_df = utils.get_collection_as_datafarme(DATABASE_NAME,TEST_COLLECTION_NAME)
        test_df  = pd.read_excel(input_file_path)
        print(test_df.columns)
        test_df.replace(to_replace="na",value=np.NAN,inplace=True)
        print(test_df.columns)
        logging.info(f'{test_df.head()}')
        modified_df = utils.extracting_new_columns(test_df,"batch_pred_data")
        logging.info(f"Loading transformer to transform dataset")
        transformer = utils.load_object(file_path=model_resolver.get_latest_transformer_path())
        test_arr = transformer.transform(modified_df)

        logging.info(f"Loading model to make prediction")
        model = utils.load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(test_arr)
        # concatinating the predicted values to test file
        # original_file = utils.get_collection_as_datafarme(DATABASE_NAME,TEST_COLLECTION_NAME)
        original_file = pd.read_excel(input_file_path)
        original_file['PredictedPrice'] = prediction
        prediction_file_name = f"Predicted_file_{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv"
        print(f"{datetime.now().strftime('%m%d%Y_%H%M%S')}")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        original_file.to_csv(prediction_file_path,index=False,header=True)
    except Exception as e:
        raise FlightFareException(e)