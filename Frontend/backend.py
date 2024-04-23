from flask import Flask, request, jsonify
import requests
import pandas as pd
from google.cloud import storage
import pickle
import logging
import time
from sklearn.preprocessing import MinMaxScaler
import gcsfs

fs = gcsfs.GCSFileSystem()

logging.basicConfig(filename='backend.log', level=logging.INFO)
# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

app = Flask(__name__)

# Define route for the root URL
@app.route('/')
def index():
    return 'Welcome to the Flask API!'

# Define route to handle CSV file upload
@app.route('/process_csv', methods=['POST'])
def process_csv_file():
    # Get CSV file from request
    csv_file = request.files['file']
    
    # Read CSV file into pandas DataFrame
    df = pd.read_csv(csv_file)
    #push inference dataframe to bucket
    inference_df_bytes = pickle.dumps(df)
    file_name = "inference_original_csv.pickle"
    bucket_name = 'aml_mlops_bucket'
    folder_name = "airflow_files"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_name}/{file_name}")
    blob.upload_from_string(inference_df_bytes, content_type='application/octet-stream')
    #wait for airflow dag2 to preprocess the inference df
    #time.sleep(30)
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
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(inf_X)
            # Convert the normalized data back to a DataFrame
            inf_X = pd.DataFrame(normalized_data, columns=inf_X.columns)
    except Exception as e:
        logging.error(f"An error occurred while loading inference_preprocessed_ddfaf_csv data: {e}")

    file_name = "'model_from_airflow.pickle'"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_name}/{file_name}")
    model_bytes = blob.download_as_string()
    model = pickle.loads(model_bytes)

    y_pred = model.predict(inf_X)
    df_with_predictions = pd.concat([df, pd.DataFrame(y_pred, columns=['Is_Laundering_Prediction'])], axis=1)
    print(df_with_predictions)
    # Save DataFrame with prediction results as CSV file
    output_csv_file = 'prediction_result.csv'
    df_with_predictions.to_csv(output_csv_file, index=False)
    
    # Filter fraudulent transactions
    fraudulent_transactions = df_with_predictions[df_with_predictions['Is_Laundering_Prediction'] == 1]
    print(fraudulent_transactions)
    # Save fraudulent transactions as CSV file
    fraudulent_transactions_csv_file = 'fraudulent_transactions.csv'
    fraudulent_transactions.to_csv(fraudulent_transactions_csv_file, index=False)
    
    # Return JSON response with download links
    response_data = {
    'df_with_predictions': df_with_predictions.to_dict(orient='records'),
    'fraudulent_transactions': fraudulent_transactions.to_dict(orient='records'),
    'entire_csv_download_link': f'/download/{output_csv_file}',
    'fraudulent_transactions_download_link': f'/download/{fraudulent_transactions_csv_file}'
}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

'''
Code steps: 

It defines a Flask application.
It sets up logging to both a file (backend.log) and the console.
It defines a route (/process_csv) to handle CSV file uploads via POST requests.
Upon receiving a CSV file, it reads it into a pandas DataFrame.
It uploads the DataFrame as a pickled object to a Google Cloud Storage bucket.
It waits for 30 seconds (you might adjust this value as needed).
It retrieves a preprocessed inference DataFrame from another location in the same bucket.
It normalizes the inference DataFrame using a MinMaxScaler.
It downloads a pickled machine learning model from the bucket.
It uses the model to make predictions on the normalized inference DataFrame.
It concatenates the predictions with the inference DataFrame.
It saves the DataFrame with prediction results as a CSV file (prediction_result.csv).
It filters fraudulent transactions based on the predictions.
It saves the fraudulent transactions as a separate CSV file (fraudulent_transactions.csv).
It returns a JSON response containing download links for both CSV files.
'''