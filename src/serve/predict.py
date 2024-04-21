from flask import Flask, jsonify, request
from google.cloud import storage
import joblib
import os
import json
from dotenv import load_dotenv
import pickle
import time
from sklearn.preprocessing import MinMaxScaler


load_dotenv()

app = Flask(__name__)

def initialize_variables():
    """
    Initialize environment variables.
    Returns:
        tuple: The project id and bucket name.
    """
    project_id="skilful-alpha-415221"
    bucket_name="aml_bucket_mlops"
    return project_id, bucket_name

def initialize_client_and_bucket(bucket_name):
    """
    Initialize a storage client and get a bucket object.
    Args:
        bucket_name (str): The name of the bucket.
    Returns:
        tuple: The storage client and bucket object.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return storage_client, bucket

def load_model(bucket, bucket_name):
    """
    Fetch and load the latest model from the bucket.
    Args:
        bucket (Bucket): The bucket object.
        bucket_name (str): The name of the bucket.
    Returns:
        _BaseEstimator: The loaded model.
    """
    latest_model_blob_name = fetch_latest_model(bucket_name)
    local_model_file_name = os.path.basename(latest_model_blob_name)
    model_blob = bucket.blob(latest_model_blob_name)
    model_blob.download_to_filename(local_model_file_name)
    model = joblib.load(local_model_file_name)
    return model

def fetch_latest_model(bucket_name, prefix="model/model_"):
    """Fetches the latest model file from the specified GCS bucket.
    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix of the model files in the bucket.
    Returns:
        str: The name of the latest model file.
    """
    # List all blobs in the bucket with the given prefix
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # Extract the timestamps from the blob names and identify the blob with the latest timestamp
    blob_names = [blob.name for blob in blobs]
    if not blob_names:
        raise ValueError("No model files found in the GCS bucket.")

    latest_blob_name = sorted(blob_names, key=lambda x: x.split('_')[-1], reverse=True)[0]

    return latest_blob_name

def preprocess_data(df):
    #push inference dataframe to bucket
    inference_df_bytes = pickle.dumps(df)
    file_name = "inference_original_csv.pickle"
    bucket_name = 'aml_bucket_mlops'
    folder_name = "airflow_files"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_name}/{file_name}")
    blob.upload_from_string(inference_df_bytes, content_type='application/octet-stream')
    #wait for airflow dag2 to preprocess the inference df
    time.sleep(30)
    #retrieve the preprocessed inference df
    inf_X = pd.DataFrame()

    try:
        # Load the train pickled data from the file into a DataFrame
        gcs_test_data_path = "gs://aml_bucket_mlops/airflow_files/inference_preprocessed_ddfaf_csv.pickle"
        with fs.open(gcs_test_data_path, 'rb') as f:
            preprocessed_inf_df = pickle.load(f).reset_index()
            inf_X = preprocessed_inf_df.drop(columns=['Is_Laundering', 'Index', 'index'])
            inf_y = preprocessed_inf_df['Is_Laundering']
            # Fit the scaler to your data and transform it
            normalized_data = scaler.fit_transform(inf_X)
            # Convert the normalized data back to a DataFrame
            inf_X = pd.DataFrame(normalized_data, columns=inf_X.columns)
    except Exception as e:
        logging.error(f"An error occurred while loading inference_preprocessed_ddfaf_csv data: {e}")

    return inf_X


@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """Health check endpoint that returns the status of the server.
    Returns:
        Response: A Flask response with status 200 and "healthy" as the body.
    """
    return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    """
    Prediction route that normalizes input data, and returns model predictions.
    Returns:
        Response: A Flask response containing JSON-formatted predictions.
    """
    request_json = request.get_json()
    request_instances = request_json['instances']

    # Parse the JSON string containing CSV data into a DataFrame
    df = pd.read_json(request_instances)

    preprocessed_infdf = preprocess_data(df)

    # Make predictions with the model
    prediction = model.predict(preprocessed_infdf)
    inf_data_with_prediction = pd.concat([preprocessed_infdf, pd.DataFrame(prediction, columns=['Is_Laundering_Prediction'])], axis=1)
    return jsonify(inf_data_with_prediction)

project_id, bucket_name = initialize_variables()
storage_client, bucket = initialize_client_and_bucket(bucket_name)
#stats = load_stats(bucket)
model = load_model(bucket, bucket_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
