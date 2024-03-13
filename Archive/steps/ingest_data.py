
import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_path: str) -> None:
        """Initialize the data ingestion class."""
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Args:
        data_path: str, path to the data file
    Returns:
        df: pd.DataFrame, the ingested data
    """
    try:
        ingest_obj = IngestData(data_path)
        df = ingest_obj.get_data()
        return df
    except Exception as e:
        logging.error("Error while ingesting data")
        raise e









# import logging

# import pandas as pd
# from zenml import step


# class IngestData:
#     """
#     Data ingestion class which ingests data from the source and returns a DataFrame.
#     """

#     def __init__(self,datapath:str) -> None:
#         """Initialize the data ingestion class."""
#         self.data_path = datapath

#     def get_data(self) -> pd.DataFrame:
#         logging.info(f"Ingesting data from")
#         return pd.read_csv(self.data_path) # We can add local path also directly here ot via bucket


# @step
# def ingest_data(data_path: str) -> pd.DataFrame:
#     """
#     Args:
#         None
#     Returns:
#         df: pd.DataFrame
#     """
#     try:
#         ingest_data = IngestData()
#         df = ingest_data.get_data()
#         return df
#     except Exception as e:
#         logging.error("Error while Ingesting data")
#         raise e

