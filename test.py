import pandas as pd
import numpy as np
from config.paths_config import *
from utils.common_functions import load_data
test_df=load_data(PROCESSED_TEST_DATA_PATH)
X_test=test_df.drop(columns=['booking_status'],axis=1)
y_test=train_df['booking_status']
print("X_test : ",X_test)
print("y_test : ",y_test.columns)

