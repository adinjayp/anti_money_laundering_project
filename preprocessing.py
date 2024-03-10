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

def initial_preprocessing(raw_data, first_timestamp):
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
        
        # Creating a pandas DataFrame
    pandas_df = pd.DataFrame(data, columns=['Index', 'From_ID', 'To_ID', 'Timestamp', 'Amount_Paid', 'Payment_Currency',
                                     'Amount_Received', 'Receiving_Currency', 'Payment_Format', 'Is_Laundering'])

    ddf = dd.from_pandas(pandas_df, npartitions=2)

    return ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict


def add_edges_to_graph(G, ddf):
    # Your functions to add edges to the graph here
    def add_edges(partition):
        for index, row in partition.iterrows():
            G.add_edge(row['From_ID'], row['To_ID'], 
                       timestamp=row['Timestamp'], 
                       amount_sent=row['Amount_Paid'], 
                       amount_received=row['Amount_Received'], 
                       received_currency=row['Receiving_Currency'], 
                       payment_format=row['Payment_Format'])

    ddf.map_partitions(add_edges).compute()
    return ddf

def create_graph(ddf):
    # Your graph creation functions here
    G = nx.DiGraph()
    ddf = add_edges_to_graph(G, ddf)
    
    return G, ddf


# input_file = "HI-Small_Trans.csv"
# raw_data = dt.fread(input_file, columns=dt.str32, fill=True)

# Convert the raw_data DataTable to a pandas DataFrame
# raw_data_df = raw_data.to_pandas()

# print("Column names:", raw_data_df.columns)
# print("First few rows of data:")
# print(raw_data_df.head())

# Now, continue with the rest of your code...

# For example:
# initial_preprocessed_ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict = initial_preprocessing(raw_data_df, first_timestamp=-1)
# G, ddf = create_graph(initial_preprocessed_ddf)
import datatable as dt
import numpy as np
from datetime import datetime
from datatable import f, join, sort
import pandas as pd
import dask.dataframe as dd
import sys
import os
from sklearn.model_selection import train_test_split

def initial_preprocessing_test(raw_data, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict):

    data = []

    def get_dict_value(name, collection):
        if name in collection:
            value = collection[name]
        else:
            value = len(collection)
            collection[name] = value
        return value

    for i in range(raw_data.nrows):
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
        
    # Creating a pandas DataFrame
    pandas_df = pd.DataFrame(data, columns=['Index', 'From_ID', 'To_ID', 'Timestamp', 'Amount_Paid', 'Payment_Currency',
                                     'Amount_Received', 'Receiving_Currency', 'Payment_Format', 'Is_Laundering'])

    ddf = dd.from_pandas(pandas_df, npartitions=2)

    return ddf, first_timestamp, currency_dict, payment_format_dict, bank_account_dict, account_dict
