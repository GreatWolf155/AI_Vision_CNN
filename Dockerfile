# CNN Vision in the cloud (Google Cloud Run): the Streamlit site and the REST API in one container.
# Keras runs on JAX here (no GPU, and far less memory than PyTorch). Deploy steps are in the README.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 KERAS_BACKEND=jax
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install --no-cache-dir -r deploy/requirements.txt

COPY . .
RUN mkdir -p logs

# Cloud Run sends the traffic to port 8080. The site's public address comes from the VISION_SITE_URL setting.
EXPOSE 8080
CMD ["python", "main.py", "--serve", "--no-ngrok", "--no-browser", "--port", "8080", "--update-every", "0"]
