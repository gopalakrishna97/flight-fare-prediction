from flightfare.entity import artifact_entity,config_entity
from flightfare.exception import FlightFareException
from flightfare.logger import logging
import os, sys
from typing import Optional
import pandas as pd
import numpy as np
from flightfare import utils
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV


class ModelTrainer:
    
    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise FlightFareException(e)

    def fine_tune(self,x,y):
        try:
            params = {
            'n_estimators' : [300, 500, 700, 1000, 2100],
            'max_depth' : [3, 5, 7, 9, 11, 13, 15],
            'max_features' : ["auto", "sqrt", "log2"],
            'min_samples_split' : [2, 4, 6, 8]
                }
            rs_rfr=RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = params, cv = 10, n_jobs=-1, verbose=3)
            rs_rfr.fit(x, y)
            best_params = rs_rfr.best_params_
            return best_params
        except Exception as e:
            raise FlightFareException(e)

    def train_model(self,x,y):
        try:
            # gb_reg =  GradientBoostingRegressor()
            # best_params = self.fine_tune(x,y)
            # logging.info(f"best_params : {best_params}")
            # {'n_estimators': 1000,
            # 'min_samples_split': 2,
            # 'max_features': 'auto',
            # 'max_depth': 13}

            # rf_reg = RandomForestRegressor(n_estimators=best_params.n_estimators,
            #                                 min_samples_split=best_params.min_samples_split,
            #                                 max_features=best_params.max_features,
            #                                 max_depth=best_params.max_depth)
            # rf_reg = RandomForestRegressor( n_estimators=500,min_samples_split= 8,max_features= 0.1 ,max_depth= 13)
            rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            # rf_reg = LinearRegression()
            rf_reg.fit(x,y)
            return rf_reg
        except Exception as e:
            raise FlightFareException(e)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]


            logging.info(f"Train the model")
            model = self.train_model(x=x_train,y=y_train)

            logging.info(f"Calculating r2_score train score")
            yhat_train = model.predict(x_train)
            r2_score_train = r2_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f"Calculating r2_score test score")
            yhat_test = model.predict(x_test)
            r2_score_test = r2_score(y_true=y_test, y_pred=yhat_test)

            logging.info(f"train score:{r2_score_train} and tests score {r2_score_test}")

            logging.info(f"Checking if our model is underfitting or not")

            if r2_score_test<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {r2_score_test}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(r2_score_train-r2_score_test)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
            r2_score_train=float(r2_score_train), r2_score_test=float(r2_score_test))
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

            return "ddvdv"
        except Exception as e:
            raise FlightFareException(e)