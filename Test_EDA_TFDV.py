#script file for Test_EDA

#!python3 -m pip install scikit-learn
#pip install tensorflow_data_validation
#pip install utils
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

#Define a class for data ingestion:

class IngestData:
    def __init__(self, aml_mlops_bucket: str) -> None:
        self.gcp_bucket_name = aml_mlops_bucket
        self.gcp_file_path1 = gs://aml_mlops_bucket/HI_Medium_Trans_1.csv
        self.gcp_file_path2 = gs://aml_mlops_bucket/HI_Small_Trans.csv

    def get_data(self) -> pd.DataFrame:
        try:
            logging.info(f"Ingesting data from GCP bucket: {self.gcp_bucket_name}")
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket_name)
            
            blob1 = bucket.blob(self.gcp_file_path1)
            raw_data_bytes1 = blob1.download_as_bytes()
            aml_test = pd.read_csv(raw_data_bytes1).astype(str)

            blob2 = bucket.blob(self.gcp_file_path1)
            raw_data_bytes2 = blob2.download_as_bytes()
            aml_train = pd.read_csv(raw_data_bytes2).astype(str)

            return {"data1":aml_test, "data2":aml_train} 
        
        except Exception as e:
            logging.error(f"Error while ingesting data from GCP: {e}")
            raise e

#Define a function for EDA:
def perform_eda(df: pd.DataFrame) -> None:
    logging.info("Starting exploratory data analysis")
    print(df.head())
    print(df.info())
    print(df.describe(include='all'))
    print(df.isna().sum())
    
    null_values_total = df.isnull().sum()
    null_rows = None
    for column in null_values_total.index:
        if null_values_total[column] > 0:  # Check for null values before assignment
            null_rows = df.loc[df[column].isnull(), :]
            logging.warning(f"Null values found in column '{column}':")
            logging.warning(null_rows.to_string())  # Use logging for null rows
    
    negative_amount_paid = df[df['Amount Paid'] < 0]
    if not negative_amount_paid.empty:
        logging.warning("Rows with negative values in 'Amount Paid' column:")
        logging.warning(negative_amount_paid.to_string())

    negative_amount_received = df[df['Amount Received'] < 0]
    if not negative_amount_received.empty:
        logging.warning("\nRows with negative values in 'Amount Received' column:")
        logging.warning(negative_amount_received.to_string())

    logging.info("Exploratory data analysis completed")

#Define a function for TFDV:    
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


#Integration with Main Script:

#Ingest data using data_dict = data_ingestor.get_data().
#Call analyze_with_tfdv(data_dict) to initiate TFDV analysis for both DataFrames.

# ... rest of your main script logic ...

#Configure Logging 
logging.basicConfig(filename='data_ingestion_and_analysis.log', level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
