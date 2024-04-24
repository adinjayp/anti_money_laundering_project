import os
import logging
import pickle
import json
from google.cloud import storage
import gcsfs

fs = gcsfs.GCSFileSystem()


def upload_file_to_gcs(dagtype, **kwargs):
    """
    Upload a file to Google Cloud Storage bucket.
    
    Parameters:
    bucket_name (str): Name of the GCS bucket.
    file_path (str): Path to the file to be uploaded.
    
    Returns:
    str: URL of the uploaded file, or None if upload fails.
    """
    G_bytes = kwargs['task_instance'].xcom_pull(task_ids='add_edges_to_graph', key='G_data')['G']
    merged_ddf_bytes = kwargs['task_instance'].xcom_pull(task_ids='merge_trans_with_gf', key='merged_ddf_bytes')

    if dagtype == 'initial':
        first_timestamp = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['first_timestamp']
        currency_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['currency_dict']
        payment_format_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['payment_format_dict']
        bank_account_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['bank_account_dict']
        account_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing', key='preprocessing_data')['account_dict']
    
    elif dagtype == 'inference':
        first_timestamp = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing_test', key='preprocessing_data')['first_timestamp']
        currency_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing_test', key='preprocessing_data')['currency_dict']
        payment_format_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing_test', key='preprocessing_data')['payment_format_dict']
        bank_account_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing_test', key='preprocessing_data')['bank_account_dict']
        account_dict = kwargs['task_instance'].xcom_pull(task_ids='initial_preprocessing_test', key='preprocessing_data')['account_dict']

    # Convert the dictionary to a JSON string
    json_account_dict = json.dumps(account_dict)
    json_currency_dict = json.dumps(currency_dict)
    json_payment_format_dict = json.dumps(payment_format_dict)
    json_bank_account_dict = json.dumps(bank_account_dict)
    json_first_timestamp = json.dumps({"first_timestamp": first_timestamp})

    ddf_file_name = "train_preprocessed_ddfaf_csv.pickle" if dagtype == 'initial' else "inference_preprocessed_ddfaf_csv.pickle"

    files_to_push = [[G_bytes,'graphaf.gpickle'],
    [json_first_timestamp, 'first_timestampaf.json'], 
    [json_currency_dict, 'currency_dictaf.json'],
    [json_payment_format_dict, 'payment_format_dictaf.json'],
    [json_bank_account_dict, 'bank_account_dictaf.json'],
    [json_account_dict, 'account_dictaf.json'],
    [merged_ddf_bytes, ddf_file_name]]

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
        
        for file in files_to_push:
            # Upload the file to the bucket
            blob = bucket.blob(f"{folder_name}/{file[1]}")
            blob.upload_from_string(file[0], content_type=content_type_finder(file[1]))

            # Construct the URL of the uploaded file
            file_url = f'gs://{bucket_name}/{folder_name}/{file[1]}'

            # Log the upload
            logging.info(f"File '{file[1]}' uploaded successfully to GCS bucket '{bucket_name}' as '{file_url}'")

        if dagtype=='inference':
            # Load the train pickled data from the file into a DataFrame
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            file_name = 'train_preprocessed_ddfaf_csv.pickle'
            blob = bucket.blob(f"{folder_name}/{file_name}")
            preprocessed_train_bytes = blob.download_as_string()
            preprocessed_train_df = pickle.loads(preprocessed_train_bytes)
            tain_and_inf_df = pd.concat([preprocessed_train_df, preprocessed_inf_df], axis=0)
            tain_and_inf_df_bytes = pickle.dumps(tain_and_inf_df)
            blob = bucket.blob(f"{folder_name}/train_preprocessed_ddfaf_csv.pickle")
            blob.upload_from_string(tain_and_inf_df_bytes, content_type='application/octet-stream')
            # Construct the URL of the uploaded file
            file_url = f'gs://{bucket_name}/{folder_name}/train_preprocessed_ddfaf_csv.pickle'
            # Log the upload
            logging.info("File trainwinf_preprocessed_ddfaf_csv uploaded successfully to GCS bucket")


        return 
    
    except Exception as e:
        # Log error if upload fails
        logging.error(f"An error occurred while uploading files to GCS bucket '{bucket_name}': {e}")
        return None

# Set up logging
logging.basicConfig(level=logging.INFO)