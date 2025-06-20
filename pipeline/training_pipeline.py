import yaml
from utils.common_functions import read_yaml
from config.paths_config import *
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTraining

if __name__=="__main__":
    ###1. Data ingestion
    data_ingestion=DataIngestion(config=read_yaml(CONFIG_PATH))
    data_ingestion.run()
    ###2. Data preprocessing
    data_preprocessor=DataPreprocessor(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        processsed_dir=Proccessed_DIR,
        config_path=CONFIG_PATH
    )
    data_preprocessor.process()

    ###3.model training
    pipeline=ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
        )
    pipeline.run()
    