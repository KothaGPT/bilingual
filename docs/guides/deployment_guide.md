# Bilingual AI Deployment Guide

This guide covers deployment options for the Bilingual AI models, including local deployment, cloud deployment, and serverless deployment. The guide has been updated to reflect the latest improvements including QLoRA fine-tuning and optimized tokenizer integration.

> **Note**: This guide assumes you have already trained your models using the updated training scripts with QLoRA support.

## Table of Contents
- [Model Deployment](#model-deployment)
  - [Loading QLoRA Fine-tuned Models](#loading-qlora-fine-tuned-models)
  - [Merging Adapters for Production](#merging-adapters-for-production)
  - [Model Quantization](#model-quantization)
- [Local Deployment](#local-deployment)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the API Server](#running-the-api-server)
  - [Environment Variables](#environment-variables)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Serverless Deployment](#serverless-deployment)
- [Performance Optimization](#performance-optimization)
  - [QLoRA-Specific Optimizations](#qlora-specific-optimizations)
  - [GPU Optimization](#gpu-optimization)
- [API Reference](#api-reference)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Serverless Deployment](#serverless-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Performance Optimization](#performance-optimization)


## Model Deployment

### Loading QLoRA Fine-tuned Models

When deploying QLoRA fine-tuned models, you'll need to load both the base model and the trained adapters. Here's how to load a QLoRA model with optimized settings:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Configuration
model_name = "mistralai/Mistral-7B-v0.1"  # Base model
peft_model_id = "path/to/your/qlora/checkpoint"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    truncation_side="left",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load QLoRA adapter
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()  # Set to evaluation mode
```

### Merging Adapters for Production

For production deployments, merge the QLoRA adapters with the base model for improved inference speed:

```python
# Merge adapters with base model
merged_model = model.merge_and_unload()

# Save the merged model
output_dir = "path/to/merged-model"
merged_model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

# Save model card
with open(f"{output_dir}/README.md", "w") as f:
    f.write(f"# Merged Model\n\nMerged from {peft_model_id}")
```

### Model Quantization

For efficient deployment, use 4-bit quantization with BitsAndBytes:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "path/to/merged-model",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Local Deployment

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- NVIDIA GPU with >=16GB VRAM (for 7B models)
- pip and virtual environment

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install "bilingual-nlp[inference]"  # For basic inference
pip install "bilingual-nlp[all]"       # For all optional dependencies

# Install QLoRA dependencies
pip install "peft>=0.4.0" "transformers>=4.30.0" "accelerate>=0.20.0" "bitsandbytes>=0.39.0"

# For Flash Attention 2.0 (recommended for faster inference)
pip install flash-attn --no-build-isolation
```

### Running the API Server

For production deployment with optimized settings:

```bash
# Start the FastAPI server with multiple workers
uvicorn bilingual.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --timeout-keep-alive 300 \
  --log-level info \
  --worker-class uvicorn.workers.UvicornWorker \
  --limit-concurrency 100 \
  --backlog 2048
```

### Environment Variables

Create a `.env` file with these recommended production settings:

```bash
# Model Configuration
MODEL_NAME=path/to/your/merged-model
TOKENIZER_NAME=path/to/your/tokenizer
MAX_INPUT_LENGTH=4096
MAX_NEW_TOKENS=1024
TEMPERATURE=0.7
TOP_P=0.95
TOP_K=50
REPETITION_PENALTY=1.1

# Performance Settings
USE_FLASH_ATTENTION_2=true
TORCH_DTYPE=bfloat16
DEVICE_MAP=auto
LOAD_IN_4BIT=true

# API Settings
API_KEY=your_secure_api_key
ENABLE_METRICS=true
LOG_LEVEL=info
RATE_LIMIT=100/1m  # 100 requests per minute
```

## Docker Deployment

### Prerequisites
- Docker 20.10+
- NVIDIA Container Toolkit (for GPU support)

### Dockerfile Example

```dockerfile
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "bilingual.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  bilingual-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=/app/models/merged-model
      - TOKENIZER_NAME=/app/models/tokenizer
      - MAX_INPUT_LENGTH=4096
      - MAX_NEW_TOKENS=1024
      - USE_FLASH_ATTENTION_2=true
      - TORCH_DTYPE=bfloat16
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
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

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use 4-bit quantization

2. **Slow Inference**
   - Enable Flash Attention 2.0
   - Use a more powerful GPU
   - Optimize input sequence length

3. **Installation Issues**
   - Ensure CUDA toolkit matches PyTorch version
   - Use the correct Python version (3.10+)
   - Check for dependency conflicts

### Getting Help

For additional support:
- Check the project's GitHub Issues
- Review the documentation
- Join our community Discord

## API Reference

### Chat Completion

```bash
POST /api/v1/chat
```

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 0.9,
  "top_k": 50,
  "language": "en"  // Optional: force response language
}
```

**Response:**
```json
{
  "response": "Hello! How can I assist you today?",
  "tokens_generated": 15,
  "model": "bilingual-7b",
  "timestamp": "2025-03-15T12:00:00Z"
}
```

#### Model Information

```bash
GET /api/v1/info
```

**Response:**
```json
{
  "model_name": "bilingual-7b",
  "model_version": "1.0.0",
  "tokenizer_name": "bilingual-tokenizer",
  "tokenizer_version": "1.0.0",
  "device": "cuda:0"
}
