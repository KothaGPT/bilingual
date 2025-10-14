# 🚀 Production Deployment Guide

## Quick Start Deployment

### Option 1: Docker Compose (Recommended for Production)

```bash
# Build and start all services
docker-compose up -d --build

# Check service status
docker-compose ps

# View logs
docker-compose logs -f bilingual-api

# Scale the API service
docker-compose up -d --scale bilingual-api=3

# Stop services
docker-compose down
```

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Start the API server
python3 -m bilingual.server --host 0.0.0.0 --port 8000 --workers 4

# Or using uvicorn directly
uvicorn bilingual.server:app --host 0.0.0.0 --port 8000 --workers 4 --reload
```

### Option 3: Production with Systemd

Create a systemd service file `/etc/systemd/system/bilingual-api.service`:

```ini
[Unit]
Description=Bilingual NLP API Server
After=network.target

[Service]
Type=simple
User=bilingual
WorkingDirectory=/opt/bilingual
Environment=PATH=/opt/bilingual/venv/bin
Environment=BILINGUAL_MODEL_DEFAULT_MODEL=t5-small
ExecStart=/opt/bilingual/venv/bin/uvicorn bilingual.server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable bilingual-api
sudo systemctl start bilingual-api
sudo systemctl status bilingual-api
```

## Health Monitoring

### Health Check Endpoints

```bash
# API Health
curl http://localhost:8000/health

# Server Status
curl http://localhost:8000/status

# Metrics (if Prometheus enabled)
curl http://localhost:8000/metrics
```

### Monitoring Setup

1. **Prometheus** (Optional)
   ```bash
   # Start monitoring stack
   docker-compose --profile monitoring up -d

   # Access Grafana
   # URL: http://localhost:3000
   # Username: admin
   # Password: admin
   ```

2. **Log Aggregation**
   ```bash
   # View application logs
   docker-compose logs -f bilingual-api

   # Or with journalctl for systemd
   sudo journalctl -u bilingual-api -f
   ```

## Scaling and Load Balancing

### Horizontal Scaling

```bash
# Scale API instances
docker-compose up -d --scale bilingual-api=5

# Or manually
docker run -d --name bilingual-api-2 \
  -p 8001:8000 \
  ghcr.io/kothagpt/bilingual:latest
```

### Load Balancer Setup (nginx example)

```nginx
upstream bilingual_backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    server_name api.bilingual.local;

    location / {
        proxy_pass http://bilingual_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://bilingual_backend/health;
    }
}
```

## Security Hardening

### SSL/TLS Setup

```bash
# Generate SSL certificate
sudo certbot --nginx -d api.bilingual.local

# Or use self-signed for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=api.bilingual.local"
```

### Firewall Configuration

```bash
# Allow API port
sudo ufw allow 8000

# Allow load balancer port
sudo ufw allow 80

# Allow SSL port
sudo ufw allow 443

# Enable firewall
sudo ufw enable
```

## Performance Optimization

### Model Optimization

```bash
# Convert models to ONNX for faster inference
python3 scripts/onnx_converter.py --model t5-small --output models/onnx/

# Quantize models for reduced memory usage
python3 scripts/onnx_converter.py --model t5-small --output models/onnx/ --quantize
```

### Caching Setup

```python
# Add Redis for caching (docker-compose)
# Update docker-compose.yml with Redis service
# Then configure in application
import redis
cache = redis.Redis(host='redis', port=6379, db=0)
```

## Backup and Recovery

### Database Backups (if using database)

```bash
# Backup Redis data
docker exec redis redis-cli SAVE

# Copy backup files
docker cp redis:/data/dump.rdb ./backups/redis-$(date +%Y%m%d).rdb
```

### Configuration Backups

```bash
# Backup configuration files
cp docker-compose.yml docker-compose.yml.backup
cp .env .env.backup
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check what's using port 8000
   sudo lsof -i :8000

   # Change port in docker-compose.yml
   # ports:
   #   - "8001:8000"
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats

   # Reduce model size or scale down
   # Use t5-small instead of t5-base
   ```

3. **Model Loading Errors**
   ```bash
   # Check model cache
   ls -la models/cache/

   # Clear cache if needed
   rm -rf models/cache/*
   ```

### Debug Mode

```bash
# Start in debug mode
BILINGUAL_LOG_LEVEL=DEBUG docker-compose up

# Or check container logs
docker-compose logs bilingual-api
```

## Production Checklist

- [ ] Environment variables configured (.env file)
- [ ] SSL/TLS certificates installed
- [ ] Load balancer configured
- [ ] Monitoring and alerting set up
- [ ] Backup strategy implemented
- [ ] Security hardening applied
- [ ] Performance testing completed
- [ ] Documentation updated
- [ ] Team training completed

## Support

For production issues:
- 📖 **Documentation**: https://bilingual.readthedocs.io
- 🐛 **Issues**: https://github.com/kothagpt/bilingual/issues
- 💬 **Discussions**: https://github.com/kothagpt/bilingual/discussions
- 📧 **Email**: support@bilingual-nlp.org
