import pandas as pd
from google.cloud import storage
import pickle
import logging

logging.basicConfig(filename='model_inferencing.log', level=logging.INFO)
# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)


def model_inference_def(**kwargs):
    """Make a prediction to a deployed custom trained model
    Args:
        project (str): Project ID
        endpoint_id (str): Endpoint ID
        instances (Union[Dict, List[Dict]]): Dictionary containing instances to predict
        location (str, optional): Location. Defaults to "us-east1".
        api_endpoint (str, optional): API Endpoint. Defaults to "us-east1-aiplatform.googleapis.com".
    """
    merged_ddf_bytes = kwargs['task_instance'].xcom_pull(task_ids='merge_trans_with_gf', key='merged_ddf_bytes')
    preprocessed_inference_df = pickle.loads(merged_ddf_bytes)
    inf_X = preprocessed_inference_df.drop(columns=['Is_Laundering', 'Index'])
    inf_y_orig = preprocessed_inference_df['Is_Laundering']

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
    inference_df_with_prediction = pd.concat([inf_X, pd.DataFrame(y_pred, columns=['Is_Laundering_Prediction'])], axis=1)
    inference_df_with_prediction_bytes = pickle.dumps(inference_df_with_prediction)
    # Upload the file to the bucket
    blob = bucket.blob(f"{folder_name}/inference_df_with_prediction.pickle")
    blob.upload_from_string(inference_df_with_prediction_bytes, content_type='application/octet-stream')
    # Log the upload
    logging.info(f"File inference_df_with_prediction uploaded successfully to GCS bucket.'")

    return None
