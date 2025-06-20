import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path=train_path
        self.test_path=test_path
        self.model_output_path=model_output_path
        
        self.light_params=LIGHT_PARAMS
        self.random_search_params=RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):
        """
        Loads the data from csv files and splits the data into training and testing data frames.
        returns:
            Train and Test Dataframes.
        """
        try:
            logger.info(f"splitting the data in to train and test part")
            logger.info(f"Loading data from {self.train_path} ")
            train_df=load_data(self.train_path)
            
            logger.info(f"loading data from {self.test_path}")
            test_df=load_data(self.test_path)

            X_train=train_df.drop(columns=['booking_status'],axis=1)
            y_train=train_df['booking_status']

            X_test=test_df.drop(columns=['booking_status'],axis=1)
            y_test=test_df['booking_status']

            logger.info(f"Data splitted scuccessfully for model training")
            return X_train,y_train,X_test,y_test
        
        except Exception as e:
            logger.error(f"Error occured while loading or splitting the data {e}")
            raise CustomException("error occured while loading or splitting the data",e)

    
    def train_lgbm(self,X_train,y_train):
        """
        Training and finding the best lgbm model thorugh hyperparamter tuning
        Args:
            X_train: train data set containing dependent features.
            y_train: train data set containing target feature.
        Return:
            returns the tunned lgbm model.
        Raises:
            CustomException: If there is any error during model training.

        """
        try:
            logger.info("Intializing the Model training")
            lgbm=lgb.LGBMClassifier(random_state=42)

            logger.info("starting RandomSearch for hyperparameter tuning")
            random_search=RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=self.light_params,
                n_iter=RANDOM_SEARCH_PARAMS['n_iter'],
                cv=RANDOM_SEARCH_PARAMS['cv'],
                verbose=RANDOM_SEARCH_PARAMS['verbose'] ,
                random_state= RANDOM_SEARCH_PARAMS['random_state'],
                scoring= RANDOM_SEARCH_PARAMS['scoring']
            )
            logger.info(f"Statring hyperparameter tuning")
            random_search.fit(X_train,y_train)
            logger.info(f"hyperparameter tunning completed successfully.")

            logger.info(f" Best parameters are : {random_search.best_params_}")
            best_lgbm_model=random_search.best_estimator_
            return best_lgbm_model
    
        except Exception as e:
            logger.error(f"Error occured duing the model building {e}")
            raise CustomException("Error occured during lgbm model training",e)
    
    def evaluate_model(self,lgbm_model,X_test,y_test):
        """
        Evaluate the  model on various metrics of accuracy,precision,recall_score,f1
        Args:
            lgbm_model: model
            X_test: test data of dependent feature.
            y_test: test data of target feature.
        Return:
            returns accuracy metrics.
        """
        try:
            logger.info(f"Evaluating our model")
            y_pred=lgbm_model.predict(X_test)

            accuracy=accuracy_score(y_test,y_pred)
            precision=precision_score(y_test,y_pred)
            recall=recall_score(y_test,y_pred)
            f1=f1_score(y_test,y_pred)

            logger.info(f"Accuracy score : {accuracy} ")
            logger.info(f"Precision score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"f1_score : {f1}")

            return {
                "Accuracy":accuracy,
                "Precision" : precision,
                "Recall": recall,
                "f1_score" : f1
            }
        except Exception as e:
            logger.error(f"error while evaluating model {e}")
            raise CustomException("Failed to evaluate model",e)
        
    def save_model(self,model):
        """
        saves the model in pickle file format.

        """
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)
            logger.info("Saving the model")
            joblib.dump(model,self.model_output_path)
            logger.info(f"model saved in {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed in saving model",e)
    
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting the model training pipeline")

                logger.info("starting mlflow experimentation")
                logger.info("logging the traning and testing dataset to MLFLOW")

                mlflow.log_artifact(self.train_path,artifact_path="datasets")
                mlflow.log_artifact(self.test_path,artifact_path="datasets")
                
                X_train,y_train,X_test,y_test=self.load_and_split_data()
                model=self.train_lgbm(X_train,y_train)
                metrics=self.evaluate_model(model,X_test,y_test)
                self.save_model(model)
                
                logger.info("Logging the model in MLFLOW")
                mlflow.log_artifact(self.model_output_path)

                logger.info("logging the params and metrics to MLFLOW")
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("model training pipeline successfully completed.")


        except Exception as e:
            logger.error(f"Error occured while runing the model pipeline {e}")
            raise CustomException("Model training pipeline failed",e)
        
if __name__=="__main__":
    pipeline=ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
        )
    pipeline.run()








        
