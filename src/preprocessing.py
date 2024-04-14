import networkx as nx
import pandas as pd
import dask.dataframe as dd
import datatable as dt
import numpy as np
from datetime import datetime
from datatable import f, join, sort
import pandas as pd
import dask.dataframe as dd
import sys
import os
from sklearn.model_selection import train_test_split
import logging
import pickle


# Configure logging
logging.basicConfig(filename='preprocessing.log', level=logging.INFO)
# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

def initial_preprocessing(first_timestamp, **kwargs):
    logging.info("Starting initial preprocessing")
    raw_data = kwargs['task_instance'].xcom_pull(task_ids='data_split', key='train_test_dfs')['train_df']
    print(raw_data.head(1))
    raw_data = dt.Frame(raw_data)
    # Your initial preprocessing functions here
    data = []

    currency_dict = {}
    payment_format_dict = {}
    bank_account_dict = {}
    account_dict = {}

    def get_dict_value(name, collection):
        if name in collection:
            value = collection[name]
        else:
            value = len(collection)
            collection[name] = value
        return value

    try:
        for i in range(raw_data.shape[0]):
            datetime_object = datetime.strptime(raw_data[i, "Timestamp"], '%Y/%m/%d %H:%M')
            timestamp = datetime_object.timestamp()
            day = datetime_object.day
            month = datetime_object.month
            year = datetime_object.year

            if first_timestamp == -1:
                start_time = datetime(year, month, day)
                first_timestamp = start_time.timestamp() - 10

            timestamp = timestamp - first_timestamp

            receiving_currency = get_dict_value(raw_data[i, "Receiving Currency"], currency_dict)
            payment_currency = get_dict_value(raw_data[i, "Payment Currency"], currency_dict)

            payment_format = get_dict_value(raw_data[i, "Payment Format"], payment_format_dict)

            from_acc_id_str = raw_data[i, "From Bank"] + raw_data[i, 2]
            from_id = get_dict_value(from_acc_id_str, account_dict)

            to_acc_id_str = raw_data[i, "To Bank"] + raw_data[i, 4]
            to_id = get_dict_value(to_acc_id_str, account_dict)

            amount_received = float(raw_data[i, "Amount Received"])
            amount_paid = float(raw_data[i, "Amount Paid"])

            is_laundering = int(raw_data[i, "Is Laundering"])

            data.append([i, from_id, to_id, timestamp, amount_paid, payment_currency, amount_received, receiving_currency,
                         payment_format, is_laundering])

        logging.info("Creating pandas DataFrame")
        pandas_df = pd.DataFrame(data, columns=['Index', 'From_ID', 'To_ID', 'Timestamp', 'Amount_Paid', 'Payment_Currency',
                                         'Amount_Received', 'Receiving_Currency', 'Payment_Format', 'Is_Laundering'])
        ddf = dd.from_pandas(pandas_df, npartitions=1)
        ddf_bytes = pickle.dumps(ddf)

        logging.info("Finished initial preprocessing")
        # Combine all the data into a dictionary
        preprocessing_data = {
            'ddf': ddf_bytes,
            'first_timestamp': first_timestamp,
            'currency_dict': currency_dict,
            'payment_format_dict': payment_format_dict,
            'bank_account_dict': bank_account_dict,
            'account_dict': account_dict
        }
        
        # Push the dictionary to XCom
        kwargs['task_instance'].xcom_push(key='preprocessing_data', value=preprocessing_data)

        return {'ddf': ddf, 'first_timestamp': first_timestamp, 'currency_dict': currency_dict, 'payment_format_dict': payment_format_dict, 'bank_account_dict': bank_account_dict, 'account_dict': account_dict}

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        return None, None, None, None, None, None
