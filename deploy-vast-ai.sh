#!/bin/bash

# Voice Server Deployment Script for Vast AI
# This script sets up the voice server on a Vast AI GPU instance

set -e  # Exit on any error

echo "🚀 Starting Voice Server Deployment on Vast AI..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
   exit 1
fi

# Update system
print_status "Updating system packages..."
apt update && apt upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    print_status "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    systemctl enable docker
    systemctl start docker
    print_success "Docker installed successfully"
else
    print_success "Docker already installed"
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_status "Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    print_success "Docker Compose installed successfully"
else
    print_success "Docker Compose already installed"
fi

# Install NVIDIA Docker runtime
if ! docker info | grep -q "nvidia"; then
    print_status "Installing NVIDIA Docker runtime..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update
    apt-get install -y nvidia-docker2
    systemctl restart docker
    print_success "NVIDIA Docker runtime installed successfully"
else
    print_success "NVIDIA Docker runtime already installed"
fi

# Check GPU availability
print_status "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_success "GPU detected"
else
    print_warning "nvidia-smi not found. GPU support may not be available."
fi

# Create application directory
APP_DIR="/opt/voice-server"
print_status "Creating application directory: $APP_DIR"
mkdir -p $APP_DIR
cd $APP_DIR

# Check if repository is already cloned
if [ ! -d ".git" ]; then
    print_status "Cloning repository..."
    # Replace with your actual repository URL
    git clone https://github.com/yourusername/voice-server.git .
else
    print_status "Repository already exists, pulling latest changes..."
    git pull
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file..."
    cat > .env << EOF
# AWS Configuration (Required for S3 uploads)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name-here
S3_FOLDER_PATH=audio-files

# Server Configuration
GRPC_PORT=50051
EOF
    print_warning "Please edit .env file with your actual AWS credentials and S3 bucket name"
else
    print_success ".env file already exists"
fi

# Create directories for volumes
print_status "Creating volume directories..."
mkdir -p models logs

# Build and start the container
print_status "Building Docker image..."
docker-compose build

print_status "Starting voice server..."
docker-compose up -d

# Wait for service to start
print_status "Waiting for service to start..."
sleep 30

# Check if service is running
if docker-compose ps | grep -q "Up"; then
    print_success "Voice server is running!"
    
    # Get container logs
    print_status "Container logs:"
    docker-compose logs --tail=20
    
    # Check GPU usage
    print_status "GPU usage:"
    nvidia-smi
    
    # Test the service
    print_status "Testing gRPC service..."
    if command -v grpcurl &> /dev/null; then
        grpcurl -plaintext -d '{"id": "test", "text": "Hello from Vast AI"}' \
            localhost:50051 audio_service.AudioService/GenerateAudio
    else
        print_warning "grpcurl not installed. Install it to test the service:"
        echo "wget https://github.com/fullstorydev/grpcurl/releases/download/v1.8.7/grpcurl_1.8.7_linux_x86_64.tar.gz"
        echo "tar -xzf grpcurl_1.8.7_linux_x86_64.tar.gz"
        echo "sudo mv grpcurl /usr/local/bin/"
    fi
    
    print_success "Deployment completed successfully!"
    print_status "Your voice server is now running on port 50051"
    print_status "You can test it using the Python client: python audio_grpc_client.py"
    
else
    print_error "Service failed to start. Check logs:"
    docker-compose logs
    exit 1
fi

# Create monitoring script
cat > /opt/voice-server/monitor.sh << 'EOF'
#!/bin/bash
echo "=== Voice Server Status ==="
echo "Container status:"
docker-compose ps
echo ""
echo "GPU usage:"
nvidia-smi
echo ""
echo "Recent logs:"
docker-compose logs --tail=10
echo ""
echo "Service health:"
curl -s http://localhost:50051 || echo "gRPC service is running"
EOF

chmod +x /opt/voice-server/monitor.sh

print_success "Monitoring script created: /opt/voice-server/monitor.sh"
print_status "Run './monitor.sh' to check server status"

# Create auto-shutdown script (optional)
cat > /opt/voice-server/auto-shutdown.sh << 'EOF'
#!/bin/bash
# Auto-shutdown after 2 hours of inactivity
echo "Server will auto-shutdown after 2 hours of inactivity"
timeout 7200 tail -f /dev/null || shutdown -h now
EOF

chmod +x /opt/voice-server/auto-shutdown.sh

print_warning "Auto-shutdown script created. Uncomment the following line in your startup script if you want auto-shutdown:"
echo "# /opt/voice-server/auto-shutdown.sh &"

print_success "🎉 Voice Server deployment completed!"
print_status "Next steps:"
print_status "1. Edit .env file with your AWS credentials"
print_status "2. Test the service: ./monitor.sh"
print_status "3. Use the Python client to generate audio" 