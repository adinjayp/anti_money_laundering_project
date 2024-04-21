FROM python:3.9-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://aml_mlops_bucket/model
ENV BUCKET_NAME=aml_mlops_bucket

COPY trainer /trainer

RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib grpcio gcsfs python-dotenv


ENTRYPOINT ["python", "-m", "trainer.train"]
