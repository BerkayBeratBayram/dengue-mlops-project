# 1) Base image
FROM python:3.10-slim

# 2) Ortam ayarları
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3) Çalışma dizini
WORKDIR /app

ENV MLFLOW_TRACKING_URI=file:/app/mlruns
ENV MLFLOW_ARTIFACT_ROOT=file:/app/mlruns

# 4) Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5) Python bağımlılıkları
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 6) Proje kodunu kopyala
COPY . .

# 7) Varsayılan komut (train çalıştırır)
CMD ["python", "-m", "src.train"]
