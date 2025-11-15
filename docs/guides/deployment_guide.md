# Deployment Guide

This guide covers various deployment options for Bilingual models, including local deployment, cloud deployment, and serverless deployment.

## Table of Contents
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Serverless Deployment](#serverless-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)

## Local Deployment

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install bilingual package
pip install bilingual-nlp

# For GPU support (recommended for production)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Running the API Server

```bash
# Start the FastAPI server
uvicorn bilingual.api:app --host 0.0.0.0 --port 8000
```

### Testing the API

```bash
# Send a test request
curl -X 'POST' \
  'http://localhost:8000/api/v1/translate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "en",
    "target_lang": "bn"
  }'
```

## Docker Deployment

### Prerequisites
- Docker
- Docker Compose (optional)

### Building the Docker Image

```bash
# Build the Docker image
docker build -t bilingual-api:latest .

# Run the container
docker run -d -p 8000:8000 --name bilingual-api bilingual-api:latest
```

### Docker Compose Example

```yaml
version: '3.8'

services:
  api:
    image: bilingual-api:latest
    build: .
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - WORKERS=4
      - LOG_LEVEL=info
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

## Kubernetes Deployment

### Prerequisites
- kubectl
- Helm (optional)
- Kubernetes cluster

### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bilingual-api
  labels:
    app: bilingual
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bilingual
  template:
    metadata:
      labels:
        app: bilingual
    spec:
      containers:
      - name: bilingual
        image: bilingual-api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service YAML

```yaml
apiVersion: v1
kind: Service
metadata:
  name: bilingual-service
spec:
  selector:
    app: bilingual
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Serverless Deployment

### AWS Lambda with API Gateway

1. Install required packages:

```bash
pip install mangum python-multipart -t ./package
```

2. Create a `lambda_handler.py`:

```python
from mangum import Mangum
from bilingual.api import app

handler = Mangum(app)
```

3. Create a deployment package:

```bash
# Create deployment package
cd package
zip -r ../deployment-package.zip .

# Add your code
cd ..
zip -g deployment-package.zip lambda_handler.py
```

4. Deploy to AWS Lambda using the AWS Console or CLI.

## Monitoring and Maintenance

### Logging

Bilingual uses Python's built-in logging module. Configure logging in your deployment:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bilingual.log')
    ]
)
```

### Metrics

Prometheus metrics are available at `/metrics` endpoint when the `PROMETHEUS_MULTIPROC_DIR` environment variable is set.

### Health Checks

- `GET /health`: Liveness probe
- `GET /ready`: Readiness probe
- `GET /metrics`: Prometheus metrics

### Scaling

For production deployments, consider:

1. Horizontal pod autoscaling (Kubernetes)
2. Database connection pooling
3. Caching frequent requests
4. CDN for static assets

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient accumulation
   - Use mixed precision training

2. **API Timeouts**:
   - Increase timeout settings
   - Optimize model loading
   - Use a larger instance type

3. **High Latency**:
   - Enable model quantization
   - Use a GPU instance
   - Implement request batching

## Support

For additional help, please open an issue on our [GitHub repository](https://github.com/bilingual-nlp/bilingual/issues).
