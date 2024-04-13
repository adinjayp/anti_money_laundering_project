import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import subprocess

# Package name to install
package_name = "imbalanced-learn"
# Execute pip install command
subprocess.call(["pip", "install", package_name])

amlm = pd.read_csv("HI-Small_Trans.csv") #read the csv file aml-medium (amlm)

amlm = amlm.drop('Timestamp', axis=1) #dropping Timestamp column as tensorflow cannot evaluate date type

class_counts = amlm['Is Laundering'].value_counts() #Counting the number of samples in each class

categorical_columns = ['Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'] #Define the columns to be label encoded
label_encoder = LabelEncoder() #creating label encoder object
# Encode the categorical columns
for column in categorical_columns:
  amlm[column] = label_encoder.fit_transform(amlm[column])
  
#Split data into training and validation sets - target class is "Is Laundering"
X = amlm.drop('Is Laundering', axis=1)
y = amlm['Is Laundering']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=22, stratify=y)
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

num_classes = len(class_counts)

# Defining the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train_over.shape[1], 1)),
    keras.layers.LSTM(32),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax"),
])
# Compiling the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
# Fit the model
history = model.fit(X_train_over, y_train_over, epochs=5, validation_data=(X_val, y_val))
# Evaluate the model on the validation data
loss, accuracy = model.evaluate(X_val, y_val)

y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)
f1 = f1_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)

#printing out the results
print("Validation loss:", loss)
print("Validation accuracy:", accuracy)
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")