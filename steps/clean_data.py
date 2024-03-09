import logging
from typing import Tuple

import pandas as pd
# from model.data_cleaning import (
#     DataCleaning,
#     DataDivideStrategy,
#     DataPreprocessStrategy,
# )
from typing_extensions import Annotated

# from zenml.steps import Output, step
from zenml import step


@step
def clean_data(
    df: pd.DataFrame) -> None:
# ) -> Tuple[
#     Annotated[pd.DataFrame, "X_train"],
#     Annotated[pd.DataFrame, "X_test"],
#     Annotated[pd.Series, "y_train"],
#     Annotated[pd.Series, "y_test"],
# ]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        # preprocess_strategy = DataPreprocessStrategy()
        # data_cleaning = DataCleaning(data, preprocess_strategy)
        # preprocessed_data = data_cleaning.handle_data()

        # divide_strategy = DataDivideStrategy()
        # data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        # X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        # logging.info("Data cleaning completed successfully")
        # return X_train, X_test, y_train, y_test
        pass
    except Exception as e:
        logging.error("Error in data cleaning step: {}".format(e))
        raise e
