import os

############################ Data Ingestion path ############################

RAW_DIR= "artifacts/raw_data"

RAW_DATA_PATH= os.path.join(RAW_DIR,'raw_data.csv')
TRAIN_DATA_PATH= os.path.join(RAW_DIR,'train.csv')
TEST_DATA_PATH=os.path.join(RAW_DIR,'test.csv')

CONFIG_PATH = "config/config.yaml"

############################ Data Processing path ############################

Proccessed_DIR = "artifacts/processed_data"
PROCESSED_TRAIN_DATA_PATH=os.path.join(Proccessed_DIR,'processed_train.csv')
PROCESSED_TEST_DATA_PATH=os.path.join(Proccessed_DIR,'processed_test.csv')

########################### MOdel Training path ###########################
MODEL_OUTPUT_PATH="artifacts/models/lgbm_model.pkl"
