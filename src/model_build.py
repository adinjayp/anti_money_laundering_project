from google.cloud import aiplatform
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import MeanSquaredError, KLDivergence
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from google.cloud import storage
import pickle
import pandas as pd
import os
import logging

logging.basicConfig(filename='model_building.log', level=logging.INFO)
# Define a stream handler to write log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)


def normalize_data(train_X, val_X):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Fit the scaler to your data and transform it
    normalized_data = scaler.fit_transform(train_X)
    train_X = pd.DataFrame(normalized_data, columns=train_X.columns)

    normalized_data = scaler.fit_transform(val_X)
    val_X = pd.DataFrame(normalized_data, columns=val_X.columns)

    return train_X, val_X

def resample_vae(train_X, train_y):
   # Define and train a Variational Autoencoder
    input_dim = train_X.shape[1]
    latent_dim = 10  # Choose an appropriate latent dimension
 
    # Define the encoder
    encoder_inputs = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(encoder_inputs)
    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)
 
    # Define the sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
 
    z = Lambda(sampling)([z_mean, z_log_var])
 
    # Define the decoder
    decoder_inputs = Input(shape=(latent_dim,))
    decoded = Dense(128, activation='relu')(decoder_inputs)
    decoded_outputs = Dense(input_dim, activation='sigmoid')(decoded)
 
    # Define the models
    encoder = Model(encoder_inputs, z_mean)
    decoder = Model(decoder_inputs, decoded_outputs)
    decoder_outputs = decoder(z)
    vae = Model(encoder_inputs, decoder_outputs)
 
    # Define the VAE loss
    reconstruction_loss = MeanSquaredError()(encoder_inputs, decoder_outputs)
    kl_loss = KLDivergence()(z_mean, z_log_var)
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam())
 
    # Train the VAE
    vae.fit(train_X, epochs=10, batch_size=32, validation_data=(val_X, None))
 
    # Generate synthetic samples for minority class
    minority_class_indices = np.where(train_y == 1)[0]
    minority_class_samples = train_X.iloc[minority_class_indices]
    synthetic_samples = vae.predict(minority_class_samples)
 
    # Combine original and synthetic samples
    X_train_balanced = np.concatenate([train_X, synthetic_samples], axis=0)
    y_train_balanced = np.concatenate([train_y, np.ones(len(synthetic_samples))], axis=0)
 
    # Optionally, you can further balance the dataset using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_balanced, y_train_balanced)
 
    # Now, you can train your classifier using X_train_resampled and y_train_resampled
 
    return X_train_resampled, y_train_resampled

def randomforestmodeling(X_train_resampled, y_train_resampled, val_X, val_y):
    # Create and fit the Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # Predict on the validation set
    y_pred = model.predict(val_X)

    accuracy = accuracy_score(val_y, y_pred)
    classification_report_metrics = classification_report(val_y, y_pred)

    return model, accuracy, classification_report_metrics


def build_model(**kwargs):

    preprocessed_train_df_bytes = kwargs['task_instance'].xcom_pull(task_ids='merge_trans_with_gf', key='merged_ddf_bytes')
    preprocessed_train_df = pickle.loads(preprocessed_train_df_bytes)

    #Download the hi_medium dataframe from the bucket
    bucket_name = "aml_mlops_bucket"
    folder_name = "airflow_files"
    file_name = "inference_preprocessed_ddfaf_csv.pickle"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_name}/{file_name}")
    preprocessed_val_df_bytes = blob.download_as_string()
    preprocessed_val_df = pickle.loads(preprocessed_val_df_bytes)

    logging.info(f"train and val retreived from GCS bucket")

    train_X = preprocessed_train_df.drop(columns=['Is_Laundering', 'Index'])
    train_y = preprocessed_train_df['Is_Laundering']
    val_X = preprocessed_val_df.drop(columns=['Is_Laundering', 'Index'])
    val_y = preprocessed_val_df['Is_Laundering']

    train_X, val_X = normalize_data(train_X, val_X)

    X_train_resampled, y_train_resampled = resample_vae(train_X, val_X)
    logging.info(f"train data resampled")

    model, accuracy, classification_report_metrics = randomforestmodeling( X_train_resampled, y_train_resampled, val_X, val_y)
    logging.info(f"modeling done")

    model_bytes = pickle.dumps(model)
    classification_report_metrics_bytes = pickle.dumps(classification_report_metrics)

    blob = bucket.blob(f"{folder_name}/'model_from_airflow.pickle'")
    blob.upload_from_string(model_bytes, content_type='application/octet-stream')

    blob = bucket.blob(f"{folder_name}/'classification_report_from_airflow.pickle'")
    blob.upload_from_string(classification_report_metrics_bytes, content_type='application/octet-stream')

    logging.info(f"Files model and classification report uploaded successfully to GCS bucket")

    return None