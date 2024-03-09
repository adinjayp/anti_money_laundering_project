import logging
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    """Train the model using the preprocessed data."""
    try:
        # Your model training code goes here
        pass
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e