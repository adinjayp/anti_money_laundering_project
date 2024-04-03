# import networkx as nx
# import pandas as pd
# import dask.dataframe as dd
# import numpy as np
# import ast
# import dask
# from dask.distributed import Client
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from ingest_data import ingest_data
from preprocessing import initial_preprocessing
from create_graph import create_graph
from data_split import data_split
from feature_Extraction import process_graph_data
from dask_handling import create_dask_dataframe
from graph_operations import merge_trans_with_gf
from upload_files_to_bucket import upload_file_to_gcs

# G = None 
# scheduler_address = 'tcp://10.128.0.5:8786'

# # Connect to the Dask cluster
# client = Client(scheduler_address)
# client.upload_file('ingest_data.py')
# client.upload_file('preprocessing.py')
# client.upload_file('pre_extraction.py')
# client.upload_file('create_graph.py')
# client.upload_file('graph_operations.py')
# client.upload_file('dask_handling.py')
# client.upload_file('add_edges_to_graph.py')
# client.upload_file('data_split.py')
# client.upload_file('feature_Extraction.py')
# client.upload_file('graph_operations.py')

default_args = {
    'owner': 'adinjay',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 11),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
} 


with DAG(
    dag_id='My_v2',
    default_args=default_args,
    description="Antimoney Laundering Project",
    start_date=datetime(2024, 3, 10, 2),
    schedule_interval="@daily"
) as dag:
    
   # Tasks will be defined here
    ingest_data_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
        dag=dag
    )
    data_split_task = PythonOperator(
        task_id='data_split',
        python_callable=data_split,
        op_kwargs={'raw_data': ingest_data_task.output},
        dag=dag
    )
    preprocess_data_task = PythonOperator(
        task_id='initial_preprocessing',
        python_callable=initial_preprocessing,
        op_kwargs={'raw_data': data_split_task.output['train_df'],'first_timestamp': -1},
        dag=dag
    )    
    create_graph_task = PythonOperator(
        task_id='create_graph',
        python_callable=create_graph,
        op_kwargs={'initial_preprocessed_ddf': preprocess_data_task.output['ddf']},  # Pass the output of extract_features_task to create_graph
        dag=dag
    )
    feature_Extraction_task = PythonOperator(
        task_id='process_graph_data',
        python_callable=process_graph_data,
        op_kwargs={'G': create_graph_task.output['G'], 'train_graph_ddf': create_graph_task.output['ddf']},  # Pass the outputs of preprocess_data_task and create_graph_task
        dag=dag
    )
    create_dask_dataframe_task = PythonOperator(
        task_id='create_dask_dataframe',
        python_callable=create_dask_dataframe,
        op_kwargs={'graph_features': feature_Extraction_task.output['graph_features']},  # Pass the output of process_graph_data_task to create_dask_dataframe
        dag=dag
    )
    merge_trans_with_gf_task = PythonOperator(
        task_id='merge_trans_with_gf',
        python_callable=merge_trans_with_gf,
        op_kwargs={'transactions_ddf': create_graph_task.output['ddf'], 'graph_features_ddf': create_dask_dataframe_task.output},  # Pass the outputs of preprocess_data_task and create_dask_dataframe_task
        dag=dag
    )

    upload_files_to_gcs_task = PythonOperator(
        task_id='upload_files_to_gcs',
        python_callable=upload_file_to_gcs,
        provide_context=True,  # Allows accessing task context
        op_kwargs={'bucket_name': 'aml_mlops_bucket' ,'file_paths': [create_graph_task.output['G'], preprocess_data_task.output['first_timestamp'], preprocess_data_task.output['currency_dict'], preprocess_data_task.output['payment_format_dict'], 
                                  preprocess_data_task.output['bank_account_dict'], preprocess_data_task.output['account_dict'], merge_trans_with_gf_task.output]},  # Define file paths here
        dag=dag
    )

    
    ingest_data_task >> data_split_task >> preprocess_data_task >> create_graph_task >> feature_Extraction_task >> create_dask_dataframe_task >> merge_trans_with_gf_task >> upload_files_to_gcs_task
