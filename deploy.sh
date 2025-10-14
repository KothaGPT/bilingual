#!/bin/bash
"""
Production deployment script for the Bilingual NLP Toolkit.

This script handles:
- Building optimized ONNX models
- Creating Docker images
- Deploying to production environments
- Setting up monitoring and logging
"""

set -e  # Exit on any error

# Configuration
PROJECT_NAME="bilingual"
DOCKER_IMAGE="${PROJECT_NAME}:latest"
DOCKER_COMPOSE_FILE="docker-compose.yml"
MODEL_DIR="models"
DATA_DIR="data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check requirements
check_requirements() {
    log_info "Checking deployment requirements..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed."
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found. Install Docker for containerized deployment."
    fi

    # Check required Python packages
    if ! python3 -c "import fastapi, uvicorn, torch, transformers" &> /dev/null; then
        log_error "Required Python packages not installed. Run: pip install -r requirements.txt"
        exit 1
    fi

    log_success "Requirements check passed"
}

# Build optimized models
build_optimized_models() {
    log_info "Building optimized ONNX models..."

    # Create models directory
    mkdir -p "${MODEL_DIR}/onnx"

    # Convert models to ONNX (if Python package is available)
    python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import bilingual as bb
    # Convert sample model
    onnx_path = bb.convert_to_onnx('t5-small', 'models/pytorch/', 'models/onnx/')
    print(f'✅ ONNX model created: {onnx_path}')
except Exception as e:
    print(f'⚠️  Could not convert models: {e}')
    print('Models will be downloaded on first use')
"

    log_success "Model optimization completed"
}

# Build Docker image
build_docker_image() {
    if command -v docker &> /dev/null; then
        log_info "Building Docker image: ${DOCKER_IMAGE}"

        # Build the image
        docker build -t "${DOCKER_IMAGE}" .

        # Tag with version if available
        if git describe --tags &> /dev/null; then
            VERSION=$(git describe --tags)
            docker tag "${DOCKER_IMAGE}" "${PROJECT_NAME}:${VERSION}"
            log_success "Docker image tagged as ${PROJECT_NAME}:${VERSION}"
        fi

        log_success "Docker image built successfully"
    else
        log_warning "Docker not available - skipping container build"
    fi
}

# Set up production environment
setup_production_env() {
    log_info "Setting up production environment..."

    # Create necessary directories
    mkdir -p logs
    mkdir -p "${DATA_DIR}/evaluations"
    mkdir -p "${MODEL_DIR}/cache"

    # Create environment file if it doesn't exist
    if [ ! -f .env ]; then
        cat > .env << EOF
# Production Environment Configuration
BILINGUAL_MODEL_DEFAULT_MODEL=t5-small
BILINGUAL_API_HOST=0.0.0.0
BILINGUAL_API_PORT=8000
BILINGUAL_EVAL_BLEU_NGRAM_ORDER=4

# Logging
LOG_LEVEL=INFO

# Monitoring (if using Prometheus)
PROMETHEUS_ENABLED=false
EOF
        log_success "Created .env file with production defaults"
    fi

    # Set proper permissions
    chmod 600 .env
    log_success "Production environment setup completed"
}

# Deploy with Docker Compose
deploy_with_docker() {
    if [ -f "${DOCKER_COMPOSE_FILE}" ] && command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        log_info "Deploying with Docker Compose..."

        # Pull latest images
        docker-compose pull

        # Start services
        docker-compose up -d

        # Wait for services to be healthy
        log_info "Waiting for services to start..."
        sleep 10

        # Check service health
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "✅ Services are running and healthy"
            log_info "🌐 API available at: http://localhost:8000"
            log_info "📚 Documentation at: http://localhost:8000/docs"
            log_info "🔍 Health check at: http://localhost:8000/health"
        else
            log_warning "⚠️  Services started but health check failed"
        fi
    else
        log_warning "Docker Compose not available - skipping container deployment"
        log_info "To run manually: python3 -m bilingual.server --host 0.0.0.0 --port 8000"
    fi
}

# Set up monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."

    # Create Prometheus configuration
    mkdir -p monitoring

    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'bilingual-api'
    static_configs:
      - targets: ['bilingual-api:8000']
    scrape_interval: 5s
    metrics_path: '/metrics'
EOF

    log_success "Monitoring configuration created"
}

# Create deployment documentation
create_deployment_docs() {
    log_info "Creating deployment documentation..."

    cat > DEPLOYMENT.md << 'EOF'
# 🚀 Production Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/bilingual-nlp/bilingual.git
cd bilingual

# Deploy with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f bilingual-api
```

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python3 -m bilingual.server --host 0.0.0.0 --port 8000

# Or using uvicorn directly
uvicorn bilingual.server:app --host 0.0.0.0 --port 8000 --workers 4
```

## Environment Configuration

Create a `.env` file for configuration:

```bash
# Model settings
BILINGUAL_MODEL_DEFAULT_MODEL=t5-small
BILINGUAL_MODEL_CACHE_DIR=models/cache

# API settings
BILINGUAL_API_HOST=0.0.0.0
BILINGUAL_API_PORT=8000

# Monitoring
PROMETHEUS_ENABLED=true
```

## Health Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Metrics (if Prometheus enabled)
```bash
curl http://localhost:8000/metrics
```

### Server Status
```bash
curl http://localhost:8000/status
```

## Scaling

### Horizontal Scaling
```bash
# Scale to multiple instances
docker-compose up -d --scale bilingual-api=3
```

### Load Balancing
Use a reverse proxy (nginx, traefik) for load balancing and SSL termination.

## Monitoring

### Prometheus + Grafana (Optional)
```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana
# URL: http://localhost:3000
# Username: admin
# Password: admin
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure sufficient memory (models can be large)
   - Check internet connection for model downloads

2. **Port Conflicts**
   - Change port in docker-compose.yml or .env file

3. **Memory Issues**
   - Reduce model size (use t5-small instead of t5-base)
   - Enable model quantization

4. **GPU Issues**
   - Ensure CUDA is properly installed
   - Check GPU memory availability

## Production Checklist

- [ ] Environment variables configured
- [ ] SSL/TLS certificates installed
- [ ] Load balancer configured
- [ ] Monitoring and alerting set up
- [ ] Backup strategy implemented
- [ ] Security hardening applied
- [ ] Performance testing completed

## API Usage Examples

See the [API Documentation](http://localhost:8000/docs) for complete reference.

## Support

For issues and questions:
- GitHub Issues: https://github.com/bilingual-nlp/bilingual/issues
- Documentation: https://bilingual.readthedocs.io
EOF

    log_success "Deployment documentation created"
}

# Main deployment function
main() {
    echo "🚀 Bilingual NLP Toolkit - Production Deployment"
    echo "=============================================="

    # Run deployment steps
    check_requirements
    build_optimized_models
    setup_production_env
    setup_monitoring
    build_docker_image
    deploy_with_docker
    create_deployment_docs

    echo ""
    log_success "🎉 Deployment completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Review DEPLOYMENT.md for detailed instructions"
    echo "2. Access the API at http://localhost:8000"
    echo "3. Check the documentation at http://localhost:8000/docs"
    echo "4. Monitor logs with: docker-compose logs -f"
    echo ""
    echo "🔧 For production: Configure SSL, load balancing, and monitoring"
}

# Run deployment if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
