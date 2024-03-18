import os
import logging
import pickle
import json
from google.cloud import storage

def upload_file_to_gcs(bucket_name, file_path):
    """
    Upload a file to Google Cloud Storage bucket.
    
    Parameters:
    bucket_name (str): Name of the GCS bucket.
    file_path (str): Path to the file to be uploaded.
    
    Returns:
    str: URL of the uploaded file, or None if upload fails.
    """
    try:
        # Initialize a Google Cloud Storage client
        storage_client = storage.Client()

        # Extract file name from the file path
        file_name = os.path.basename(file_path)

        # Specify the content type based on file extension
        content_type = 'application/octet-stream'
        if file_name.endswith('.json'):
            content_type = 'application/json'
        elif file_name.endswith('.csv'):
            content_type = 'text/csv'

        # Specify the blob object
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Read file data
        with open(file_path, 'rb') as file:
            file_data = file.read()

        # Upload the file to the bucket
        blob.upload_from_string(file_data, content_type=content_type)

        # Construct the URL of the uploaded file
        file_url = f'gs://{bucket_name}/{file_name}'

        # Log the upload
        logging.info(f"File '{file_path}' uploaded successfully to GCS bucket '{bucket_name}' as '{file_name}'")

        return 
    
    except Exception as e:
        # Log error if upload fails
        logging.error(f"An error occurred while uploading file '{file_path}' to GCS bucket '{bucket_name}': {e}")
        return None

# Set up logging
logging.basicConfig(level=logging.INFO)

# Example usage:
bucket_name = 'aml_mlops_bucket'
file_path_graph = 'graph.gpickle'
file_path_account_dict = 'account_dict.json'
file_path_currency_dict = 'currency_dict.json'
file_name_payment_format_dict = "payment_format_dict.json"
file_name_bank_account_dict = "bank_account_dict.json"
file_name_first_timestamp = "first_timestamp.json"
file_name_merged_ddf = "merged_ddf.csv"

# Upload files to GCS bucket
graph_url = upload_file_to_gcs(bucket_name, file_path_graph)
account_dict_url = upload_file_to_gcs(bucket_name, file_path_account_dict)
currency_dict_url = upload_file_to_gcs(bucket_name, file_path_currency_dict)
payment_format_dict_url = upload_file_to_gcs(bucket_name, file_name_payment_format_dict)
bank_account_dict_url = upload_file_to_gcs(bucket_name, file_name_bank_account_dict)
first_timestamp_dict_url = upload_file_to_gcs(bucket_name, file_name_first_timestamp)
merged_ddf_url = upload_file_to_gcs(bucket_name,file_name_merged_ddf)

# if graph_url and account_dict_url and currency_dict_url:
#     print("Files uploaded successfully to GCS bucket:")
#     print("Graph:", graph_url)
#     print("Account Dictionary:", account_dict_url)
#     print("Currency Dictionary:", currency_dict_url)
#     print("Payment Format Dictionary:", payment_format_dict_url)
#     print("Bank Account Dictionary:", bank_account_dict_url)
#     print("First Timestamp Dictionary:", first_timestamp_dict_url)

# else:
#     print("Failed to upload files to GCS bucket. Check logs for details.")
