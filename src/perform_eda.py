import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd
from sklearn.model_selection import train_test_split
#from utils import add_extra_rows
#from tensorflow_metadata.proto.v0 import schema_pb2
import logging
import pandas as pd
#from tensorflow.python.data.ops import dataset_ops  # TFDV functionality
#from tensorflow_metadata.proto.v0 import schema_pb2
from google.cloud import storage  
import datetime# For accessing GCP buckets

def perform_eda(**kwargs) -> None:
    logging.info("Starting exploratory data analysis")
    df = kwargs['task_instance'].xcom_pull(task_ids='read_validation_data', key='test_data_from_cloud')['test_df']
    logging.info("test_df.info: %s", str(df.info()))
    logging.info("test_df.describe: %s", str(df.describe(include='all')))
    logging.info("Number of null values: %s", str(df.isna().sum()))
    logging.info("First few rows: %s", str(df.head()))
    
    null_values_total = df.isnull().sum()
    null_rows = None
    for column in null_values_total.index:
        if null_values_total[column] > 0:  # Check for null values before assignment
            null_rows = df.loc[df[column].isnull(), :]
            logging.warning(f"Null values found in column '{column}':")
            logging.warning(null_rows.to_string())  # Use logging for null rows
    
    negative_amount_paid = df[df['Amount Paid'].astype(float) < 0]
    if not negative_amount_paid.empty:
        logging.warning("Rows with negative values in 'Amount Paid' column:")
        logging.warning(negative_amount_paid.to_string())

    negative_amount_received = df[df['Amount Received'].astype(float) < 0]
    if not negative_amount_received.empty:
        logging.warning("\nRows with negative values in 'Amount Received' column:")
        logging.warning(negative_amount_received.to_string())

    logging.info("Exploratory data analysis completed")
    return 
