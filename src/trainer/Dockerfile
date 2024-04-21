FROM python:3.9-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://aml_bucket_mlops/model
ENV BUCKET_NAME=aml_bucket_mlops

COPY trainer /trainer

RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib grpcio gcsfs python-dotenv imbalanced-learn tensorflow==2.15.0


ENTRYPOINT ["python", "-m", "trainer.train"]
