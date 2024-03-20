import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd
from sklearn.model_selection import train_test_split
#from utils import add_extra_rows
from tensorflow_metadata.proto.v0 import schema_pb2
import logging
import pandas as pd
from tensorflow.python.data.ops import dataset_ops  # TFDV functionality
from tensorflow_metadata.proto.v0 import schema_pb2
from google.cloud import storage  
import datetime# For accessing GCP buckets

def perform_eda(df: pd.DataFrame) -> None:
    logging.info("Starting exploratory data analysis")
    print(df.head())
    print(df.info())
    print(df.describe(include='all'))
    print(df.isna().sum())
    
    null_values_total = df.isnull().sum()
    null_rows = None
    for column in null_values_total.index:
        if null_values_total[column] > 0:  # Check for null values before assignment
            null_rows = df.loc[df[column].isnull(), :]
            logging.warning(f"Null values found in column '{column}':")
            logging.warning(null_rows.to_string())  # Use logging for null rows
    
    negative_amount_paid = df[df['Amount Paid'] < 0]
    if not negative_amount_paid.empty:
        logging.warning("Rows with negative values in 'Amount Paid' column:")
        logging.warning(negative_amount_paid.to_string())

    negative_amount_received = df[df['Amount Received'] < 0]
    if not negative_amount_received.empty:
        logging.warning("\nRows with negative values in 'Amount Received' column:")
        logging.warning(negative_amount_received.to_string())

    logging.info("Exploratory data analysis completed")
    return 
