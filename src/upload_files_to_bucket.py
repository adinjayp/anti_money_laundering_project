import os
import logging
import pickle
import json
from google.cloud import storage

def upload_file_to_gcs(**kwargs):
    """
    Upload a file to Google Cloud Storage bucket.
    
    Parameters:
    bucket_name (str): Name of the GCS bucket.
    file_path (str): Path to the file to be uploaded.
    
    Returns:
    str: URL of the uploaded file, or None if upload fails.
    """
    G_bytes = kwargs['task_instance'].xcom_pull(task_ids='add_edges_to_graph', key='G_data')['G']

    first_timestamp = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['first_timestamp']
    currency_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['currency_dict']
    payment_format_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['payment_format_dict']
    bank_account_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['bank_account_dict']
    account_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['account_dict']

    merged_ddf = kwargs['task_instance'].xcom_pull(task_ids='merge_trans_with_gf', key='merged_ddf')
    # Convert DataFrame to CSV format
    merged_ddf_csv = merged_ddf.to_csv(index=False)

    files_to_push = [[G_bytes,'airflow_files/graphaf.gpickle'],
    [first_timestamp, 'first_timestampaf.json'], 
    [currency_dict, 'currency_dictaf.json'],
    [payment_format_dict, 'payment_format_dictaf.json'],
    [bank_account_dict, 'bank_account_dictaf.json'],
    [account_dict, 'account_dictaf.json'],
    [merged_ddf_csv, "train_preprocessed_ddfaf.csv"]]

    bucket_name = 'aml_mlops_bucket'
    folder_name = "airflow_files"

    def content_type_finder(file_name):
        # Specify the content type based on file extension
        if file_name.endswith('.json'):
            content_type = 'application/json'
        elif file_name.endswith('.csv'):
            content_type = 'text/csv'
        else:
            content_type = 'application/octet-stream'
        return content_type


    try:
        # Initialize a Google Cloud Storage client
        storage_client = storage.Client()
        # Specify the blob object
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        for file in files_to_push:
            # Upload the file to the bucket
            blob.upload_from_string(file[0], content_type=content_type_finder(file[1]))

            # Construct the URL of the uploaded file
            file_url = f'gs://{bucket_name}/{folder_name}/{file[1]}'

            # Log the upload
            logging.info(f"File '{file_path}' uploaded successfully to GCS bucket '{bucket_name}' as '{file_name}'")

    return 
    
    except Exception as e:
        # Log error if upload fails
        logging.error(f"An error occurred while uploading file '{file_path}' to GCS bucket '{bucket_name}': {e}")
        return None

# Set up logging
logging.basicConfig(level=logging.INFO)