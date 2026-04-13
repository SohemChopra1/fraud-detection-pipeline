#Python 3.11 slim image
FROM python:3.11-slim


RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#directory
WORKDIR /app

#python installations 
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#project files
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

#exposes port
EXPOSE 8000

#starts API with 0.0.0.0
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]