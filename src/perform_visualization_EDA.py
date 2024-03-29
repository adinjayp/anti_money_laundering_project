import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd
from sklearn.model_selection import train_test_split
#from utils import add_extra_rows
from tensorflow_metadata.proto.v0 import schema_pb2
import logging
import pandas as pd
from tensorflow.python.data.ops import dataset_ops  # TFDV functionality
from tensorflow_metadata.proto.v0 import schema_pb2
from google.cloud import storage  
import datetime# For accessing GCP buckets



def analyze_with_tfdv(df1: pd.DataFrame, df2: pd.DataFrame, aml_mlops_bucket: str, output_folder: str = "tfdv_visualizations") -> None:
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
    bucket = storage_client.bucket(aml_mlops_bucket)
    blob = bucket.blob(f"{aml_mlops_bucket}/{output_folder}/{filename}")
    blob.upload_from_string(fig.to_string_io(), content_type="image/png")
    logging.info(f"Visualization saved: gs://{aml_mlops_bucket}/{output_folder}/{filename}")

    # Infer schema from the statistics
    schema = tfdv.infer_schema(statistics=df1_stats)

    # Display the inferred schema
    tfdv.display_schema(schema)

    # Validation step (optional)
    tfdv.validate_statistics(statistics=df1_stats, schema=schema)

    logging.info("TFDV analysis completed")

  except Exception as e:
    logging.error(f"Error during TFDV analysis: {e}")