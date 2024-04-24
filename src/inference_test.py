import datatable as dt
import networkx as nx
import pandas as pd
import dask.dataframe as dd
import numpy as np
import ast
import dask
from dask.distributed import Client
from google.cloud import storage
import pickle
import json
from datetime import datetime

# Address of the Dask scheduler
scheduler_address = 'tcp://10.128.0.5:8786'

# Connect to the Dask cluster
client = Client(scheduler_address)
client.upload_file('feature_extraction.py')
client.upload_file('graph_operations.py')
client.upload_file('preprocessing.py')

# GET G FROM BUCKET

# Initialize a Google Cloud Storage client
storage_client = storage.Client()

# Specify the name of your GCP bucket
bucket_name = 'aml_mlops_bucket'

# Specify the name of the file containing the serialized graph
file_name = 'graph.gpickle'

# Download the serialized graph from the bucket
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_name)
graph_bytes = blob.download_as_string()

# Deserialize the graph using pickle
G = pickle.loads(graph_bytes)

# first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict FROM BUCKET!!

# Specify the name of the files in the bucket
file_names = ["account_dict.json", "currency_dict.json", "payment_format_dict.json", "bank_account_dict.json", "first_timestamp.json"]

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

# Test set prep
print("begin test set prep")

gcs_bucket_path = "gs://aml_mlops_bucket/"
raw_data_pandas = pd.read_csv(gcs_bucket_path + 'HI_Medium_Trans_1.csv').astype(str)
test_dt = dt.Frame(raw_data_pandas).head()

# Check for null values in the datatable Frame
null_values_exist = (test_dt.countna().to_numpy().any())

# If null values exist, delete the transactions containing null values
if null_values_exist:
    # Extract rows with null values
    # Print the transactions containing null values before deletion
    print("Transactions containing null values:")
    print(test_dt[dt.f[:].isna().any(), :])
    # Delete rows with null values
    test_dt = test_dt[dt.f[:].isna().sum() == 0, :]

# Confirm if null values are removed
print("Null values removed:", not test_dt.countna().to_numpy().any())

test_initial_preprocessed_ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict = initial_preprocessing_test(test_dt, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict)


G, test_graph_ddf = add_edges_to_graph(G, test_initial_preprocessed_ddf)
print(G)
unique_nodes_test = list(set(test_graph_ddf['From_ID']).union(test_graph_ddf['To_ID']))

#append unique nodes whenever new accounts from test set come up
unique_nodes_dd_test = dd.from_pandas(pd.DataFrame(unique_nodes_test, columns=['Node']), npartitions=2)

graph_features_test = unique_nodes_dd_test.map_partitions(lambda df: df.apply(lambda row: extract_features(G, row['Node']), axis=1))
#graph_features_test = unique_nodes_dd_test.map_partitions(lambda df: df.apply(lambda row: {key: str(value) for key, value in extract_features(G, row['Node']).items()}, axis=1))

# Convert each string to a dictionary
dicts_test = [ast.literal_eval(str(string_data)) for string_data in graph_features_test]

# Create a list of lists containing the dictionary values for each entry
list_of_lists_test = [list(data_dict.values()) for data_dict in dicts_test]

# Create a DataFrame from the list of lists
lists_df_test = pd.DataFrame(list_of_lists_test, columns=dicts_test[0].keys())

# Convert specific columns to the desired data types
convert_dtype = {'Node': 'int64', 'degree': 'int64', 'in_degree': 'int64', 'out_degree': 'int64', 'clustering_coefficient': 'float64', 'degree_centrality': 'float64'}
graph_features_df_test = lists_df_test.astype(convert_dtype)
graph_features_ddf_test = dd.from_pandas(graph_features_df_test, npartitions=2)

preprocessed_test_df = merge_trans_with_gf(test_graph_ddf, graph_features_ddf_test)
print(preprocessed_test_df.head())

print("test data is preprocessed. next step - ML Prediction")

inf_X = preprocessed_test_df.drop(columns=['Is_Laundering', 'Index'])
inf_y_orig = preprocessed_test_df['Is_Laundering']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Fit the scaler to your data and transform it
normalized_data = scaler.fit_transform(inf_X)
inf_X = pd.DataFrame(normalized_data, columns=inf_X.columns)


#Download the hi_medium dataframe from the bucket
bucket_name = "aml_mlops_bucket"
folder_name = "airflow_files"
file_name = "'model_from_airflow.pickle'"

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(f"{folder_name}/{file_name}")
model_bytes = blob.download_as_string()
model = pickle.loads(model_bytes)

y_pred = model.predict(inf_X)
inference_df_with_prediction = pd.concat([raw_data_pandas, pd.DataFrame(y_pred, columns=['Is_Laundering_Prediction'])], axis=1)

print("Fraudulent transactions head: \n", inference_df_with_prediction[inference_df_with_prediction['Is_Laundering_Prediction'] == 1].head())
print("Total number of fraud transactions: ", len(inference_df_with_prediction[inference_df_with_prediction['Is_Laundering_Prediction'] == 1]))

inference_df_with_prediction_bytes = pickle.dumps(inference_df_with_prediction)
# Upload the file to the bucket
blob = bucket.blob(f"{folder_name}/inference_df_with_prediction.pickle")
blob.upload_from_string(inference_df_with_prediction_bytes, content_type='application/octet-stream')
# Log the upload
logging.info(f"File inference_df_with_prediction uploaded successfully to GCS bucket.'")


# UPDATING BUCKET WITH LATEST G AND OTHER DICTS
#code to push G and other files to cloud VM

# Initialize a Google Cloud Storage client
storage_client = storage.Client()

# Serialize the graph to a bytes object
graph_bytes = pickle.dumps(G)

# Specify the name of your GCP bucket
bucket_name = 'aml_mlops_bucket'

# Specify the name for the file in the bucket
file_name = 'graph.gpickle'

# Upload the serialized graph to the bucket
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_name)
blob.upload_from_string(graph_bytes, content_type='application/octet-stream')

print(f'Graph saved to gs://{bucket_name}/{file_name}')

# Convert the dictionary to a JSON string
json_account_dict = json.dumps(account_dict)
json_currency_dict = json.dumps(currency_dict)
json_payment_format_dict = json.dumps(payment_format_dict)
json_bank_account_dict = json.dumps(bank_account_dict)
json_first_timestamp = json.dumps({"first_timestamp": first_timestamp})

# Specify the name of the file to be saved in the bucket
file_name_account_dict = "account_dict.json"
file_name_currency_dict = "currency_dict.json"
file_name_payment_format_dict = "payment_format_dict.json"
file_name_bank_account_dict = "bank_account_dict.json"
file_name_first_timestamp = "first_timestamp.json"

# Define the blob object
blob_data_dict = bucket.blob(file_name_account_dict)
blob_currency_dict = bucket.blob(file_name_currency_dict)
blob_payment_format_dict = bucket.blob(file_name_payment_format_dict)
blob_bank_account_dict = bucket.blob(file_name_bank_account_dict)
blob_first_timestamp = bucket.blob(file_name_first_timestamp)


# Upload the JSON data to the bucket
blob_data_dict.upload_from_string(json_account_dict)
blob_currency_dict.upload_from_string(json_currency_dict)
blob_payment_format_dict.upload_from_string(json_payment_format_dict)
blob_bank_account_dict.upload_from_string(json_bank_account_dict)
blob_first_timestamp.upload_from_string(json_first_timestamp)


print("Data uploaded successfully to GCS bucket:", bucket_name)