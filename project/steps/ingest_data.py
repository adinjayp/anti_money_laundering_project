
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
        gcs_bucket_path = "/home/adiax/project7/anti_money_laundering_project/data/HI_Small_Trans.csv.dvc"

    def get_data(self) -> pd.DataFrame:
        logging.info(f"Ingesting data from Google Cloud Storage")
        raw_data_pandas = pd.read_csv(self.gcs_bucket_path).astype(str)
        return raw_data_pandas

def ingest_data() -> dt.Frame:
    """
    Args:
        data_path: str, path to the data file
    Returns:
        df: pd.DataFrame, the ingested data
    """
    try:
        ingest_obj = IngestData()
        df = ingest_obj.get_data()
        raw_data = dt.Frame(df)
        return raw_data
    except Exception as e:
        logging.error("Error while ingesting data")
        raise e

