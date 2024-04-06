import logging
import pandas as pd
import dask.dataframe as dd
import datatable as dt

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        self.gcs_bucket_path = "gs://aml_mlops_bucket/HI_Small_Trans.csv"

    def get_data(self) -> pd.DataFrame:
        logging.info(f"Ingesting data from Google Cloud Storage")
        raw_data_pandas = pd.read_csv(self.gcs_bucket_path).astype(str)
        return raw_data_pandas.head(10)

def ingest_data(**kwargs) -> dt.Frame:
    """
    Args:
        data_path: str, path to the data file
    Returns:
        df: pd.DataFrame, the ingested data
    """
    try:
        logging.info("Starting data ingestion")
        ingest_obj = IngestData()
        df = ingest_obj.get_data()
        raw_data = df
        #raw_data = dt.Frame(df)
        kwargs['task_instance'].xcom_push(key='raw_data', value=raw_data)
        logging.info("Data ingestion completed")
        return raw_data
    except Exception as e:
        logging.error("Error while ingesting data")
        logging.exception(e)
        raise e

# Configure logging
logging.basicConfig(filename='data_ingestion.log', level=logging.INFO)

# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

