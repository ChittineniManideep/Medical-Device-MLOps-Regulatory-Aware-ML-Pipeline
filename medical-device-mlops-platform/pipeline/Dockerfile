FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_model.py drift_monitoring.py ./

# Non-root user — standard hardening practice, more relevant than usual here
# given the container will handle (simulated) patient-derived training data
RUN useradd -m mlops && chown -R mlops:mlops /app
USER mlops

ENV MLFLOW_TRACKING_URI=""

CMD ["python", "train_model.py"]
