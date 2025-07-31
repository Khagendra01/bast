# Voice Server - Vast AI GPU Deployment Guide

This is a gRPC-based voice server that generates audio from text using TTS models. It's designed to run on GPU instances for optimal performance.

## Features

- Text-to-Speech (TTS) audio generation using Coqui TTS
- gRPC server for high-performance communication
- AWS S3 integration for audio file storage
- GPU acceleration support
- Background audio generation capabilities

## Prerequisites

- Vast AI account with GPU credits
- AWS S3 bucket and credentials (optional but recommended)
- Docker (for containerized deployment)

## Deployment Options

### Option 1: Direct Deployment on Vast AI

1. **Create a Vast AI Instance:**
   - Go to [Vast AI Console](https://console.vast.ai/)
   - Select a GPU instance with:
     - CUDA support (RTX 4090, A100, V100, etc.)
     - At least 16GB RAM
     - Ubuntu 20.04 or 22.04
   - Choose a machine with good network bandwidth

2. **Connect to Your Instance:**
   ```bash
   ssh root@YOUR_INSTANCE_IP
   ```

3. **Install Dependencies:**
   ```bash
   # Update system
   apt update && apt upgrade -y
   
   # Install Python 3.10
   apt install -y python3.10 python3.10-venv python3.10-dev
   
   # Install CUDA and PyTorch dependencies
   apt install -y nvidia-cuda-toolkit
   
   # Install system dependencies for audio processing
   apt install -y ffmpeg libsndfile1-dev portaudio19-dev
   ```

4. **Clone and Setup:**
   ```bash
   # Clone your repository
   git clone YOUR_REPO_URL
   cd voice-server
   
   # Create virtual environment
   python3.10 -m venv venv
   source venv/bin/activate
   
   # Install Python dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Environment Configuration:**
   ```bash
   # Create .env file
   cat > .env << EOF
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=us-east-1
   S3_BUCKET_NAME=your-bucket-name
   S3_FOLDER_PATH=audio-files
   GRPC_PORT=50051
   EOF
   ```

6. **Run the Server:**
   ```bash
   # Start the gRPC server
   python audio_grpc_server.py
   ```

### Option 2: Docker Deployment (Recommended)

1. **Create Dockerfile:**
   ```dockerfile
   FROM nvidia/cuda:11.8-devel-ubuntu20.04
   
   # Set environment variables
   ENV DEBIAN_FRONTEND=noninteractive
   ENV PYTHONUNBUFFERED=1
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       python3.10 \
       python3.10-venv \
       python3.10-dev \
       python3-pip \
       ffmpeg \
       libsndfile1-dev \
       portaudio19-dev \
       git \
       && rm -rf /var/lib/apt/lists/*
   
   # Set working directory
   WORKDIR /app
   
   # Copy requirements first for better caching
   COPY requirements.txt .
   
   # Install Python dependencies
   RUN pip3 install --upgrade pip
   RUN pip3 install -r requirements.txt
   
   # Copy application code
   COPY . .
   
   # Generate gRPC code
   RUN python3 -m grpc_tools.protoc \
       --proto_path=protos \
       --python_out=. \
       --grpc_python_out=. \
       protos/audio_service.proto
   
   # Expose gRPC port
   EXPOSE 50051
   
   # Create startup script
   RUN echo '#!/bin/bash\npython3 audio_grpc_server.py' > /app/start.sh
   RUN chmod +x /app/start.sh
   
   CMD ["/app/start.sh"]
   ```

2. **Create docker-compose.yml:**
   ```yaml
   version: '3.8'
   
   services:
     voice-server:
       build: .
       ports:
         - "50051:50051"
       environment:
         - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
         - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
         - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
         - S3_BUCKET_NAME=${S3_BUCKET_NAME}
         - S3_FOLDER_PATH=${S3_FOLDER_PATH}
         - GRPC_PORT=50051
       volumes:
         - ./models:/app/models  # Optional: mount models directory
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```

3. **Deploy on Vast AI:**
   ```bash
   # On your Vast AI instance
   docker-compose up -d
   ```

### Option 3: Using Vast AI's Docker Template

1. **Create a deployment script:**
   ```bash
   #!/bin/bash
   
   # Install Docker if not present
   if ! command -v docker &> /dev/null; then
       curl -fsSL https://get.docker.com -o get-docker.sh
       sh get-docker.sh
   fi
   
   # Install Docker Compose
   curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   chmod +x /usr/local/bin/docker-compose
   
   # Clone repository
   git clone YOUR_REPO_URL
   cd voice-server
   
   # Create .env file
   cat > .env << EOF
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=us-east-1
   S3_BUCKET_NAME=your-bucket-name
   S3_FOLDER_PATH=audio-files
   EOF
   
   # Build and run
   docker-compose up -d
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key for S3 uploads | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for S3 uploads | Required |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `S3_BUCKET_NAME` | S3 bucket name for audio storage | Required |
| `S3_FOLDER_PATH` | S3 folder path within bucket | `audio-files` |
| `GRPC_PORT` | gRPC server port | `50051` |

### GPU Requirements

- **Minimum:** 8GB VRAM (RTX 3070, RTX 3080)
- **Recommended:** 16GB+ VRAM (RTX 4090, A100, V100)
- **CUDA Version:** 11.8 or higher

## Testing the Deployment

1. **Test gRPC Connection:**
   ```bash
   # Install grpcurl for testing
   wget https://github.com/fullstorydev/grpcurl/releases/download/v1.8.7/grpcurl_1.8.7_linux_x86_64.tar.gz
   tar -xzf grpcurl_1.8.7_linux_x86_64.tar.gz
   sudo mv grpcurl /usr/local/bin/
   
   # Test the service
   grpcurl -plaintext -d '{"id": "test", "text": "Hello world"}' \
     localhost:50051 audio_service.AudioService/GenerateAudio
   ```

2. **Test with Python Client:**
   ```bash
   python audio_grpc_client.py
   ```

## Monitoring and Logs

1. **Check Server Status:**
   ```bash
   # Check if server is running
   netstat -tlnp | grep 50051
   
   # Check GPU usage
   nvidia-smi
   
   # Check logs
   docker logs voice-server  # if using Docker
   ```

2. **Performance Monitoring:**
   ```bash
   # Monitor GPU usage
   watch -n 1 nvidia-smi
   
   # Monitor system resources
   htop
   ```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce batch size in TTS model
   - Use smaller TTS models
   - Increase GPU memory

2. **gRPC Connection Issues:**
   - Check firewall settings
   - Verify port 50051 is open
   - Check server logs for errors

3. **S3 Upload Failures:**
   - Verify AWS credentials
   - Check S3 bucket permissions
   - Verify network connectivity

### Performance Optimization

1. **GPU Optimization:**
   ```python
   # In audio_grpc_server.py, add:
   torch.backends.cudnn.benchmark = True
   torch.backends.cudnn.deterministic = False
   ```

2. **Memory Management:**
   ```python
   # Clear GPU cache periodically
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```

## Cost Optimization

1. **Choose Right GPU:**
   - RTX 4090: Good performance/cost ratio
   - A100: Best performance, higher cost
   - V100: Good for inference, lower cost

2. **Auto-shutdown:**
   ```bash
   # Add to startup script
   echo "Server will auto-shutdown after 2 hours of inactivity"
   timeout 7200 tail -f /dev/null || shutdown -h now
   ```

## Security Considerations

1. **Network Security:**
   - Use VPN or SSH tunneling
   - Implement authentication for gRPC
   - Restrict access to specific IPs

2. **AWS Security:**
   - Use IAM roles instead of access keys
   - Enable S3 bucket encryption
   - Use VPC endpoints for S3

## Scaling

For high-traffic scenarios:

1. **Load Balancing:**
   - Use nginx or HAProxy
   - Implement health checks
   - Use multiple GPU instances

2. **Caching:**
   - Cache generated audio files
   - Use Redis for session management
   - Implement request deduplication

## Support

For issues or questions:
- Check server logs: `docker logs voice-server`
- Monitor GPU usage: `nvidia-smi`
- Test connectivity: `telnet localhost 50051`