from flightfare.utils import load_object,save_object
from flightfare.exception import FlightFareException
from flightfare.logger import logging
from flightfare.predictor import ModelResolver
from flightfare.entity.config_entity import ModelPusherConfig
from flightfare.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact,ModelEvaluationArtifact







class ModelPusher:
    def __init__(self,model_pusher_config:ModelPusherConfig,
                        data_transformation_artifact:DataTransformationArtifact,
                        model_trainer_artifact:ModelTrainerArtifact,
                        model_eval_artifact:ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_eval_artifact=model_eval_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise FlightFareException(e)


    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            # load objects
            logging.info("loading transformer and model objects")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)


            # saving in artifact-->timestamp-->model_pusher dir
            logging.info("saving models to modelpusher directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path,obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path,obj=model)

            if self.model_eval_artifact.is_model_accepted:
                # saving in  saved_model dir
                logging.info('saving model in saved_model directory')
                transformer_path = self.model_resolver.get_latest_save_transformer_path()
                model_path = self.model_resolver.get_latest_save_model_path()
                save_object(file_path=transformer_path,obj=transformer)
                save_object(file_path=model_path,obj=model)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                                                        saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"model pusher artifact : {model_pusher_artifact}")
            return model_pusher_artifact
                
        except Exception as e:
            raise FlightFareException(e)


