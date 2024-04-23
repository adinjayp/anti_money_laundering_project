from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import gcsfs
from google.cloud import storage
fs = gcsfs.GCSFileSystem()
import pickle


def predict_custom_trained_model(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """Make a prediction to a deployed custom trained model
    Args:
        project (str): Project ID
        endpoint_id (str): Endpoint ID
        instances (Union[Dict, List[Dict]]): Dictionary containing instances to predict
        location (str, optional): Location. Defaults to "us-central1".
        api_endpoint (str, optional): API Endpoint. Defaults to "us-central1-aiplatform.googleapis.com".
    """
    
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

storage_client = storage.Client()
bucket = storage_client.bucket("aml_bucket_mlops")
blob = bucket.blob('hi_medium_df.pickle')
hi_medium_df_bytes = blob.download_as_string()
test_df = pickle.loads(hi_medium_df_bytes)
test_df = test_df.astype(str)
df_json = test_df.to_json(orient='records')

predict_custom_trained_model(
    project="287941977155",
    endpoint_id="2835198484359938048",
    location="us-central1",
    instances=df_json
)
