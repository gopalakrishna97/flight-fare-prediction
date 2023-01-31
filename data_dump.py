import pymongo
import pandas as pd
import json

from flightfare.db_config import mongo_client

DATA_FILE_PATH="Data_Train.xlsx"
DATABASE_NAME="flightfare"
COLLECTION_NAME="flightdata"

if __name__=="__main__":
    df = pd.read_excel(DATA_FILE_PATH)
    print(f"Rows and Columns: {df}")

    # convert DataFrame to JSON , to dump data in mongoDB
    df.reset_index(drop=True,inplace=True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    # insert json_record to mongoDB
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)