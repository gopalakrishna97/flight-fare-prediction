import pymongo
import pandas as pd
import json

from flightfare.db_config import mongo_client

DATA_FILE_PATH="Data_Train.xlsx"
TEST_FILE_PATH = "Test_set.xlsx"
DATABASE_NAME="flightfare"
COLLECTION_NAME="flightdata"
TEST_COLLECTION_NAME = "flighttestdata"

if __name__=="__main__":
    # train_df = pd.read_excel(DATA_FILE_PATH)
    test_df = pd.read_excel(TEST_FILE_PATH)

    # print(f"Rows and Columns: {train_df}")
    print(f"Rows and Columns: {test_df}")

    # convert DataFrame to JSON , to dump data in mongoDB
    # train_df.reset_index(drop=True,inplace=True)
    test_df.reset_index(drop=True,inplace=True)

    # train_json_record = list(json.loads(train_df.T.to_json()).values())
    test_json_record = list(json.loads(test_df.T.to_json()).values())

    # print(train_json_record[0])
    print(test_json_record[0])

    # insert json_record to mongoDB
    # mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(train_json_record)
    mongo_client[DATABASE_NAME][TEST_COLLECTION_NAME].insert_many(test_json_record)
