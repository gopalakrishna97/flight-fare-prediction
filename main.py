from flightfare.pipeline.training_pipeline import start_training_pipeline
from flightfare.pipeline.batch_prediction import batch_prediction

input_file_path = "Test_set.xlsx"
if __name__=="__main__":
    try:
        batch_prediction(input_file_path=input_file_path)
        # start_training_pipeline()
    except Exception as e:
        print(e)    