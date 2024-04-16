import logging
import pickle
import json
from google.cloud import storage
import pandas as pd
import datatable as dt
#from dask.distributed import Client

def download_data_from_bucket():
    try:
        bucket_name='aml_mlops_bucket'
        folder_name = "airflow_files"
        #scheduler_address='tcp://10.128.0.5:8786'
        # Connect to the Dask cluster
        #client = Client(scheduler_address)
        #client.upload_file('feature_extraction.py')
        #client.upload_file('graph_operations.py')
        #client.upload_file('preprocessing.py')

        # GET G FROM BUCKET
        # Initialize a Google Cloud Storage client
        storage_client = storage.Client()

        # Specify the name of the file containing the serialized graph
        file_name = 'graphaf.gpickle'

        # Download the serialized graph from the bucket
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"{folder_name}/{file_name}")
        graph_bytes = blob.download_as_string()

        # Deserialize the graph using pickle
        #G = pickle.loads(graph_bytes)

        logging.info("Successfully downloaded and deserialized graph from bucket.")

        # first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict FROM BUCKET!!
        # Specify the name of the files in the bucket
        file_names = ["account_dict.json", "currency_dict.json", "payment_format_dict.json",
                      "bank_account_dict.json", "first_timestamp.json"]

        # Initialize empty dictionaries to store the data
        account_dict = {}
        currency_dict = {}
        payment_format_dict = {}
        bank_account_dict = {}
        first_timestamp_dict = {}

        # Loop through each file and download its contents
        for file_name in file_names:
            # Get the blob object
            blob = storage_client.bucket(bucket_name).blob(file_name)
            # Download the file's contents as a string
            file_contents = blob.download_as_string()
            # Decode the bytes to a string and parse the JSON data
            if file_name == "account_dict.json":
                account_dict = json.loads(file_contents.decode('utf-8'))
            elif file_name == "currency_dict.json":
                currency_dict = json.loads(file_contents.decode('utf-8'))
            elif file_name == "payment_format_dict.json":
                payment_format_dict = json.loads(file_contents.decode('utf-8'))
            elif file_name == "bank_account_dict.json":
                bank_account_dict = json.loads(file_contents.decode('utf-8'))
            elif file_name == "first_timestamp.json":
                first_timestamp_dict = json.loads(file_contents.decode('utf-8'))
                first_timestamp = int(first_timestamp_dict['first_timestamp'])

        logging.info("Successfully downloaded and parsed dictionaries from bucket.")

        # Read data from GCS bucket
        gcs_bucket_path = "gs://aml_mlops_bucket/"
        raw_data_pandas = pd.read_csv(gcs_bucket_path + 'HI_Medium_Trans_1.csv').astype(str)
        test_df = raw_data_pandas.head(25)
        logging.info("test_df head: %s", str(test_df.head()))

        logging.info("Successfully read data from GCS bucket.")

        test_data_from_cloud = {
            'G_bytes': G_bytes,
            'test_df': test_df,
            'first_timestamp': first_timestamp,
            'currency_dict': currency_dict,
            'payment_format_dict': payment_format_dict,
            'bank_account_dict': bank_account_dict,
            'account_dict': account_dict
        }

        logging.info("test_df head from xcom df: %s", str(test_data_from_cloud['test_df'].head()))

        # Push the dictionary to XCom
        kwargs['task_instance'].xcom_push(key='test_data_from_cloud', value=test_data_from_cloud)

        return {'G': G, 'first_timestamp': first_timestamp, 'currency_dict': currency_dict, 'payment_format_dict': payment_format_dict,'bank_account_dict': bank_account_dict, 'test_df': test_df}
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None, None, None, None, None, None


# Configure logging
logging.basicConfig(filename='bucket_download.log', level=logging.INFO)
# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)