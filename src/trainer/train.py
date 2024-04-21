from google.cloud import storage
from datetime import datetime
import pytz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import gcsfs
import os
from dotenv import load_dotenv
import numpy as np
import pickle
import pandas as pd
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
 
# Load environment variables
load_dotenv()
 
# Initialize variables
fs = gcsfs.GCSFileSystem()
storage_client = storage.Client()
bucket_name ="aml_mlops_bucket"
MODEL_DIR ="gs://aml_mlops_bucket/model"
 
def load_data(gcs_train_data_path):
    """
    Loads the training data from Google Cloud Storage (GCS).
    
    Parameters:
    gcs_train_data_path (str): GCS path where the training data CSV is stored.
    
    Returns:
    DataFrame: A pandas DataFrame containing the training data.
    """
 
    # Specify the path to the pickle file
    print("Load data is called")
    print("gcs_train_data_path: ", gcs_train_data_path)
    val_pickle_file_path = 'gs://aml_mlops_bucket/airflow_files/inference_preprocessed_ddfaf_csv.pickle'
 
    # Load the train pickled data from the file into a DataFrame
    with fs.open(gcs_train_data_path, 'rb') as f:
        preprocessed_train_df = pickle.load(f).reset_index()

    # Load the inference pickled data from the file into a DataFrame
    with fs.open(val_pickle_file_path, 'rb') as f:
        preprocessed_val_df = pickle.load(f).reset_index()


    return preprocessed_train_df, preprocessed_val_df
 
def data_transform(df_train, df_val):
    """
    Transforms the data by setting a datetime index, and splitting it into
    training and validation sets. It also normalizes the features.
    
    Parameters:
    df (DataFrame): The DataFrame to be transformed.
    
    Returns:
    tuple: A tuple containing normalized training features, test features,
           normalized training labels, and test labels.
    """

 
    # Splitting the data into training and validation sets (80% training, 20% validation)
    train_X = df_train.drop(columns=['Is_Laundering', 'Index', 'index'])
    train_y = df_train['Is_Laundering']
    val_X = df_val.drop(columns=['Is_Laundering', 'Index', 'index'])
    val_y = df_val['Is_Laundering']
 
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
 
    # Fit the scaler to your data and transform it
    normalized_data = scaler.fit_transform(train_X)
    # Convert the normalized data back to a DataFrame
    train_X = pd.DataFrame(normalized_data, columns=train_X.columns)
 
    # Fit the scaler to your data and transform it
    normalized_data = scaler.fit_transform(val_X)
    # Convert the normalized data back to a DataFrame
    val_X = pd.DataFrame(normalized_data, columns=val_X.columns)
 
    #Assuming train_X, train_y, val_X, val_y are already defined
 
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
 
    return X_train_resampled, val_X, y_train_resampled, val_y
 
 
def train_model(X_train, y_train):
    """
    Trains a Random Forest Regressor model on the provided data.
    
    Parameters:
    X_train (DataFrame): The training features.
    y_train (Series): The training labels.
    
    Returns:
    RandomForestRegressor: The trained Random Forest model.
    """
    # Create and fit the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    return rf_classifier
 
def save_and_upload_model(model, local_model_path, gcs_model_path):
    """
    Saves the model locally and uploads it to GCS.
    
    Parameters:
    model (RandomForestRegressor): The trained model to be saved and uploaded.
    local_model_path (str): The local path to save the model.
    gcs_model_path (str): The GCS path to upload the model.
    """
    # Save the model locally
    joblib.dump(model, local_model_path)
 
    # Upload the model to GCS
    with fs.open(gcs_model_path, 'wb') as f:
        joblib.dump(model, f)
 
def main():
    """
    Main function to orchestrate the loading of data, training of the model,
    and uploading the model to Google Cloud Storage.
    """
    # Load and transform data
    gcs_train_data_path = "gs://aml_mlops_bucket/airflow_files/train_preprocessed_ddfaf_csv.pickle"
    df_train, df_val = load_data(gcs_train_data_path)
    print(df_train.head())
    X_train, X_test, y_train, y_test = data_transform(df_train, df_val)
 
    # Train the model
    model = train_model(X_train, y_train)
 
    # Save the model locally and upload to GCS
    edt = pytz.timezone('US/Eastern')
    current_time_edt = datetime.now(edt)
    version = current_time_edt.strftime('%Y%m%d_%H%M%S')
    local_model_path = "model.pkl"
    gcs_model_path = f"{MODEL_DIR}/model_{version}.pkl"
    print(gcs_model_path)
    save_and_upload_model(model, local_model_path, gcs_model_path)
 
if __name__ == "__main__":
    main()