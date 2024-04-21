import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd
#from utils import add_extra_rows
from tensorflow_metadata.proto.v0 import schema_pb2
import logging
import pandas as pd
from tensorflow.python.data.ops import dataset_ops  # TFDV functionality
from google.cloud import storage  
import datetime
import gcsfs
import pickle

# For accessing GCP buckets
fs = gcsfs.GCSFileSystem()

def analyze_with_tfdv(**kwargs) -> None:
  """
  Analyzes two DataFrames using TFDV and generates visualizations.

  Args:
      df1: The first Pandas DataFrame for analysis.
      df2: The second Pandas DataFrame for analysis (assumed to be for comparison).
      aml_mlops_bucket: The name of the GCP bucket to store visualizations.
      output_folder: Optional subfolder within the bucket for visualizations (default: "tfdv_visualizations").
  """
  try:
    logging.info("Starting TFDV analysis")

    bucket_name = 'aml_mlops_bucket'
    output_folder = 'EDA_TFDV_testdata_viz'
    folder_name = "airflow_files"

    # GET TRAIN DF1 FROM BUCKET
    # For accessing GCP buckets
    fs = gcsfs.GCSFileSystem()
    storage_client = storage.Client()

    file_name = 'train_preprocessed_ddfaf_csv.pickle'
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_name}/{file_name}")
    df1_bytes = blob.download_as_string()
    df1 = pickle.loads(df1_bytes)  # Adjusted to use pickle.loads() instead of pickle.load()
    df1 = df1.reset_index()  # Reset the index
    logging.info("Successfully downloaded and deserialized train df from bucket.")

    # with fs.open("gs://aml_mlops_bucket/airflow_files/train_preprocessed_ddfaf_csv.pickle", 'rb') as f:
    #    df1 = pickle.load(f).reset_index()
    logging.info("Successfully downloaded and deserialized train df from bucket.")

    df2 = kwargs['task_instance'].xcom_pull(task_ids='read_validation_data', key='test_data_from_cloud')['test_df']

    #checking for null values
    null_values_total_df1 = df1.isnull().sum()
    null_rows_df1 = None
    for column in null_values_total_df1.index:
            if null_values_total_df1[column] > 0:  # Check for null values before assignment
                null_rows_df1 = df1.loc[df1[column].isnull(), :]
                logging.warning(f"Null values found in column '{column}' in df1:")
                logging.warning(null_rows_df1.to_string())  # Use logging for null rows
        
    negative_amount_paid_df1 = df1[df1['Amount Paid'] < 0]
    if not negative_amount_paid_df1.empty:
            logging.warning("Rows with negative values in 'Amount Paid' column in df1:")
            logging.warning(negative_amount_paid_df1.to_string())

    negative_amount_received_df1 = df1[df1['Amount Received'] < 0]
    if not negative_amount_received_df1.empty:
            logging.warning("\nRows with negative values in 'Amount Received' column in df1:")
            logging.warning(negative_amount_received_df1.to_string())

        # Additional code snippet for checking null values and negative values for df2
    null_values_total_df2 = df2.isnull().sum()
    null_rows_df2 = None
    for column in null_values_total_df2.index:
            if null_values_total_df2[column] > 0:  # Check for null values before assignment
                null_rows_df2 = df2.loc[df2[column].isnull(), :]
                logging.warning(f"Null values found in column '{column}' in df2:")
                logging.warning(null_rows_df2.to_string())  # Use logging for null rows
        
    negative_amount_paid_df2 = df2[df2['Amount Paid'] < 0]
    if not negative_amount_paid_df2.empty:
            logging.warning("Rows with negative values in 'Amount Paid' column in df2:")
            logging.warning(negative_amount_paid_df2.to_string())

    negative_amount_received_df2 = df2[df2['Amount Received'] < 0]
    if not negative_amount_received_df2.empty:
            logging.warning("\nRows with negative values in 'Amount Received' column in df2:")
            logging.warning(negative_amount_received_df2.to_string())

    # Generate statistics for both DataFrames
    df1_stats = tfdv.generate_statistics_from_dataframe(df1)
    df2_stats = tfdv.generate_statistics_from_dataframe(df2)

    # Visualize dataset statistics for each DataFrame
    fig = tfdv.visualize_statistics(df1_stats)
    tfdv.visualize_statistics(df2_stats)

    # Generate a unique timestamp for the visualization filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Visualize and save comparisons (assuming names for DataFrames)
    tfdv.visualize_statistics(
        lhs_statistics=df1_stats,
        rhs_statistics=df2_stats,
        lhs_name=df1.name if hasattr(df1, 'name') else "DataFrame 1",  # Use name attribute if available
        rhs_name=df2.name if hasattr(df2, 'name') else "DataFrame 2"
    )
    filename = f"comparison_{timestamp}.png"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{bucket_name}/{output_folder}/{filename}")
    blob.upload_from_string(fig.to_string_io(), content_type="image/png")
    logging.info(f"Visualization saved: gs://{bucket_name}/{output_folder}/{filename}")

    # Infer schema from the statistics
    schema = tfdv.infer_schema(statistics=df1_stats)

    # Display the inferred schema
    tfdv.display_schema(schema)

    # Validation step (optional)
    tfdv.validate_statistics(statistics=df1_stats, schema=schema)

    logging.info("TFDV analysis completed")

  except Exception as e:
    logging.error(f"Error during TFDV analysis: {e}")