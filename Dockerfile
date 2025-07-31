FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

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
    wget \
    curl \
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

# Create models directory
RUN mkdir -p /app/models

# Expose gRPC port
EXPOSE 50051

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting Voice Server..."\n\
echo "CUDA available: $(python3 -c "import torch; print(torch.cuda.is_available())")"\n\
echo "GPU count: $(python3 -c "import torch; print(torch.cuda.device_count())")"\n\
echo "GPU name: $(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\")")"\n\
python3 audio_grpc_server.py\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import grpc; import audio_service_pb2_grpc; \
    channel = grpc.insecure_channel('localhost:50051'); \
    stub = audio_service_pb2_grpc.AudioServiceStub(channel); \
    print('Health check passed')" || exit 1

CMD ["/app/start.sh"] 