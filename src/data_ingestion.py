import os 
import sys
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from google.cloud import storage
from sklearn.model_selection import train_test_split
from utils.common_functions import read_yaml
from config.paths_config import *


logger=get_logger(__name__)

class DataIngestion:
    """
    Class for data ingestion, including downloading data from Google Cloud Storage,
    splitting the dataset into training and testing sets, and saving them locally.
    """
    def __init__(self,config):
        self.config=config['data_ingestion']
        self.bucket_name=self.config['bucket_name']
        self.file_name=self.config['bucket_file_name']
        self.train_test_ratio=self.config['train_ratio']

        os.makedirs(RAW_DIR,exist_ok=True)
        logger.info(f"Data ingestion started with {self.bucket_name} bucket and file {self.file_name}")
    def download_csv_from_gcp(self):
        """
        Downloads the csv file from Google Cloud Storage to the local path.
        
        Raises:
            CustomException: If the error occurs in downloading the file from GCP.
        """
        try:
            if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
                raise CustomException("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. Please set it to your GCP service account key JSON file.", None)
            client=storage.Client()
            bucket=client.bucket(self.bucket_name)
            blob=bucket.blob(self.file_name)
            
            blob.download_to_filename(RAW_DATA_PATH)
            logger.info(f"Data downloaded successfully from {self.bucket_name} bucket to {RAW_DATA_PATH}")
        except Exception as e:
            logger.error(f"Error downloading data from GCP : {e}")
            raise CustomException("Error while downloading the data from GCP",e)
    
    def split_data(self):
        """
        Splits the dataset into training and testing sets and saves them locally.
        """
        try:
            logger.info(f"starting the data splitting process")
            #load the raw data
            raw_Data=pd.read_csv(RAW_DATA_PATH)
            

            #split the data into training and testing sets
            train_data, test_data = train_test_split(
                raw_Data,
                test_size=1 - self.train_test_ratio,
                random_state=42
            )
            logger.info(f"Raw data successfully split into train and test data with ratio ({self.train_test_ratio})")
            #save the train and test data to the local path
            train_data.to_csv(TRAIN_DATA_PATH, index=False)
            test_data.to_csv(TEST_DATA_PATH,index=False)
            logger.info(f"Train and test data saved successfully to {TRAIN_DATA_PATH} and {TEST_DATA_PATH}")

        except Exception as e:
            logger.error(f"Error occurred while splitting the data: {e}")
            raise CustomException("Error while splitting the data",e)
        
    def run(self):
        """
        Runs the data ingestion process by downloading the data from GCP,
        splitting the data into training and testing sets, and saving them locally.
        Raises:
            CustomException: If there is any error during the data ingestion process.
        """
        try:
            logger.info("starting the data ingestion process")
            #download the data from GCP
            self.download_csv_from_gcp()

            #split the data into training and testing sets
            self.split_data()

            logger.info("Data ingestion process completed successfully")

        except Exception as e:
            logger.error(f"CustomException: {str(e)}")
            raise CustomException("Error during the data ingestion process", e)
        finally:
            logger.info("Data ingestion process finished")

if __name__=="__main__":
    data_ingestion=DataIngestion(config=read_yaml(CONFIG_PATH))
    data_ingestion.run()


