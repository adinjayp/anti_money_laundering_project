FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY serve/predict.py /app/

# Install Flask and google-cloud-storage
RUN pip install Flask google-cloud-storage joblib scikit-learn grpcio gcsfs python-dotenv logging pandas

ENV AIP_STORAGE_URI=gs://aml_bucket_mlops/model
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=9000
ENV BUCKET_NAME=aml_bucket_mlops
ENV PROJECT_ID=skilful-alpha-415221


# Run serve.py when the container launches
ENTRYPOINT ["python", "predict.py"]
