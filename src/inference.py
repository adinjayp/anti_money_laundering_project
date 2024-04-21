from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-east1",
    api_endpoint: str = "us-east1-aiplatform.googleapis.com",
):
    """Make a prediction to a deployed custom trained model
    Args:
        project (str): Project ID
        endpoint_id (str): Endpoint ID
        instances (Union[Dict, List[Dict]]): Dictionary containing instances to predict
        location (str, optional): Location. Defaults to "us-east1".
        api_endpoint (str, optional): API Endpoint. Defaults to "us-east1-aiplatform.googleapis.com".
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


predict_custom_trained_model(
    project="497741562136",
    endpoint_id="1974151137439252480",
    location="us-east1",
    instances= {
            "PT08.S1(CO)": 0.651435622,
            "NMHC(GT)": 0.154088375,
            "C6H6(GT)": 0.198980682,
            "PT08.S2(NMHC)": 0.406771327,
            "NOx(GT)": 0.608617903,
            "PT08.S3(NOx)": 0.340316247,
            "NO2(GT)": 0.604898307,
            "PT08.S4(NO2)": 0.354166125,
            "PT08.S5(O3)": 0.670583278,
            "T": 0.080010396,
            "RH": 0.098031541,
            "AH": 0.146104239
        }
)