import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
from utils.common_functions import load_data,read_yaml
from config.paths_config import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger=get_logger(__name__)

class DataPreprocessor:
    def __init__(self,train_path,test_path,processsed_dir,config_path):
        """
        Initializes the DataPreprocessor with paths for training and testing data,
        processed data directory, and configuration file.
        
        Args:
            train_path (str): Path to the training data CSV file.
            test_path (str): Path to the testing data CSV file.
            processsed_dir (str): Directory where processed data will be saved.
            config_path (str): Path to the configuration YAML file.
        """
        self.train_path=train_path
        self.test_path=test_path
        self.processed_dir=processsed_dir
        self.config_path=read_yaml(config_path)
        
        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self,df):
        """
        Preprocesses the input DataFrame by handling missing values, encoding categorical features,
        and balancing the dataset using SMOTE.
        
        Args:
            df (pd.DataFrame): The input DataFrame to preprocess.
        
        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        try:
            logger.info("starting our data preprocessing step")

            logger.info("Dropping the unwanted columns and duplicates")
            df.drop(columns=['Booking_ID'],inplace=True,axis=1)
            df.drop_duplicates(inplace=True)
            
            logger.info("assigning the categorical and numerical columns from the config yaml file")
            categorical_cols=self.config_path["data_processing"]['categorical_features']
            numerical_cols=self.config_path["data_processing"]['numerical_features']

            logger.info("Label encoding the categorical columns")
            label_encoder=LabelEncoder()
            mappings={}
            for col in categorical_cols:
                df[col]=label_encoder.fit_transform(df[col])
                mappings[col]={label:code for label,code in zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_))}
            logger.info("Label mappings are:")
            for col,mapping in mappings.items():
                logger.info(f"{col}: {mapping}")
            
            logger.info("handling skewness in numerical columns")
            threshold=self.config_path['data_processing']['skew_threshold']
            skewness=df[numerical_cols].skew()
            # skewness=df[numerical_cols].apply(lambda x: x.skew())
            for col in numerical_cols:
                if skewness[col]>threshold:
                    df[col]=np.log1p(df[col])
            return df
        except Exception as e:
            logger.error(f"Error while doing data processing: {e}")
            raise CustomException("Error in doing data processing", e)
    def balance_data(self,df):
        """
        Balances the dataset using SMOTE to handle class imbalance.
        
        Args:
            df (pd.DataFrame): The input DataFrame to balance.
        
        Returns:
            pd.DataFrame: The balanced DataFrame.
        """
        try:
            logger.info("handling imbalanced dataset using SMOTE")
            X=df.drop(columns=['booking_status'],axis=1)
            y=df['booking_status']

            smote=SMOTE(random_state=42)
            X_res,y_res=smote.fit_resample(X,y)

            balanced_df=pd.DataFrame(X_res,columns=X.columns)
            balanced_df['booking_status']=y_res

            logger.info("Data balancing completed successfully")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error while balancing the data: {e}")
            raise CustomException("Error in balancing the data", e)

    def feature_selection(self,df):
        """
        Selects important features using Random Forest Regressor.
        
        Args:
            df (pd.DataFrame): The input DataFrame for feature selection.
        
        Returns:
            pd.DataFrame: The DataFrame with selected features.
        """
        try:
            logger.info("starting the feature selection process")
            X=df.drop(columns=['booking_status'],axis=1)
            y=df['booking_status']

            model=RandomForestRegressor(random_state=42)
            model.fit(X,y)

            feature_importances=model.feature_importances_
            feature_importance_df=pd.DataFrame({
                'Feature':X.columns,
                'Importance':feature_importances
            })
            sorted_features=feature_importance_df.sort_values(by='Importance',ascending=False)
            num_of_features_to_select=self.config_path['data_processing']['no_of_features']
            top_10_features=sorted_features['Feature'].head(num_of_features_to_select).values
            top_10_df=df[list(top_10_features)+['booking_status']]

            logger.info(f"feature selction completed successfully, selected features: {top_10_features}")
            return top_10_df
        
        except Exception as e:
            logger.error(f"Error while selecting features: {e}")
            raise CustomException("Error in feature selection", e)
        
    def save_data(self,df,filepath):
        """
        Saves the DataFrame to a CSV file.
        
        Args:
            df (pd.DataFrame): The DataFrame to save.
            filepath (str): The path where the DataFrame will be saved.
        """
        try:
            logger.info(f" Saving processed data in processed directory")
            df.to_csv(filepath, index=False)
            logger.info(f"Data saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error while saving data to {filepath}: {e}")
            raise CustomException(f"Error in saving data to {filepath}", e)
    
    def process(self):
        """
        Main method to run the data preprocessing steps: loading data, preprocessing,
        balancing, feature selection, and saving the processed data.
        
        Raises:
            CustomException: If there is any error during the data processing.
        """
        try:
            logger.info(f"Loading the data from raw direcotory")
            train_data=load_data(self.train_path)
            test_data=load_data(self.test_path)
            logger.info("Data loaded successfully")

            #preprocess of training and testing data
            processed_train=self.preprocess_data(train_data)
            processed_test=self.preprocess_data(test_data)

            # Balance the training data
            balanced_train=self.balance_data(processed_train)
            balanced_test=self.balance_data(processed_test)

            #Feature selection
            selected_features_train=self.feature_selection(balanced_train)
            selected_features_test=balanced_test[selected_features_train.columns]    

            #save the processed data
            self.save_data(selected_features_train,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(selected_features_test,PROCESSED_TEST_DATA_PATH)

            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException("Error in data preprocessing", e)

if __name__ =="__main__":
    data_preprocessor=DataPreprocessor(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        processsed_dir=Proccessed_DIR,
        config_path=CONFIG_PATH
    )
    data_preprocessor.process()
    
        