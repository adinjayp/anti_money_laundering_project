import logging
from sklearn.model_selection import train_test_split    
import datatable as dt
import pandas as pd
from google.cloud import storage
import os
import logging
from upload_files_to_bucket import upload_file_to_gcs


# Configure logging
logging.basicConfig(filename='data_split.log', level=logging.INFO)
# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

def data_split(**kwargs):
    logging.info("Starting data splitting")
    raw_data = kwargs['task_instance'].xcom_pull(task_ids='ingest_data_task', key='raw_data')
    try:
        train_df, test_df = train_test_split(raw_data, test_size=0.2, random_state=42, stratify=raw_data['Is Laundering'])
        train_dt = dt.Frame(train_df[:10000])
        test_dt = dt.Frame(test_df[:2000])

        upload_file_to_gcs('aml_mlops_bucket', test_dt)
        
        logging.info("Data splitting finished")
        return {'train_df': train_dt.to_pandas(), 'test_df': test_dt.to_pandas()}

    except Exception as e:
        logging.error(f"An error occurred during data splitting: {e}")
        return None, None
