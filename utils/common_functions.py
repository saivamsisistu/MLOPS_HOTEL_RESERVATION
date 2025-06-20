import os
import pandas  as pd
import yaml
from src.logger import get_logger
from src.custom_exception import CustomException

logger= get_logger(__name__)
def read_yaml(file_path):
    """
    Reads a YAML file and returns its content as a dictionary.
    
    Args:
        file_path (str): The path to the YAML file.
        
    Returns:
        dict: The content of the YAML file.
        
    Raises:
        CustomException: If there is an error reading the file.
    """
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
        return content
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {e}")
        raise CustomException(f"Error reading YAML file {file_path}",e) 
    

def load_data(file_path):
    """
    loads data from csv file
    args:
        file_path(str): path to the csv file
    returns:
        pandas.DataFrame: DataFrame containing the data from the CSV file.
    Raises:
        CustomException: If there is an error reading the file.
    
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        df=pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}:{e}")
        raise CustomException(f"Error loading data from {file_path}", e)

