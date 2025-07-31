import grpc
import logging
from typing import Optional, Tuple
import os
from dotenv import load_dotenv

# Import generated protobuf classes
import audio_service_pb2
import audio_service_pb2_grpc

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioGRPCClient:
    """Async gRPC client for audio generation service"""
    
    def __init__(self, server_address: str = "localhost:50051"):
        """
        Initialize the gRPC client
        Args:
            server_address: Address of the gRPC server (default: localhost:50051)
        """
        self.server_address = server_address
        self.channel = None
        self.stub = None
        
    async def connect(self):
        """Establish async connection to the gRPC server"""
        try:
            self.channel = grpc.aio.insecure_channel(self.server_address)
            self.stub = audio_service_pb2_grpc.AudioServiceStub(self.channel)
            # Optionally, test the connection
            logger.info(f"✅ Connected to gRPC server at {self.server_address}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to gRPC server: {e}")
            return False
    
    async def disconnect(self):
        """Close the gRPC connection"""
        if self.channel:
            await self.channel.close()
            logger.info("🔌 Disconnected from gRPC server")
    
    async def generate_audio(self, request_id: str, text: str, language: Optional[str] = None) -> Tuple[bool, str, str]:
        """
        Generate audio from text via async gRPC
        Args:
            request_id: Unique identifier for the request
            text: Text to convert to audio
        Returns:
            Tuple of (success, s3_url, error_message)
        """
        if not self.stub:
            if not await self.connect():
                return False, "", "Failed to connect to gRPC server"
        try:
            # Create request
            request = audio_service_pb2.GenerateAudioRequest(
                id=language,
                text=text,
            )
            # Make async gRPC call
            logger.info(f"🎵 Making gRPC call for ID: {request_id}")
            logger.info(f"📝 Text length: {len(text)} characters")
            response = await self.stub.GenerateAudio(request)
            if response.success:
                logger.info(f"✅ Successfully generated audio for ID: {request_id}")
                if response.s3_url:
                    logger.info(f"📁 Audio uploaded to S3: {response.s3_url}")
                return True, response.s3_url, response.error_message
            else:
                logger.error(f"❌ Audio generation failed for ID: {request_id}: {response.error_message}")
                return False, "", response.error_message
        except grpc.aio.AioRpcError as e:
            error_msg = f"gRPC error: {e.code()}: {e.details()}"
            logger.error(f"❌ {error_msg}")
            return False, "", error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, "", error_msg

# Async convenience function for easy usage
async def generate_audio_simple(request_id: str, text: str, server_address: str = "localhost:50051") -> Tuple[bool, str, str]:
    """
    Simple async function to generate audio from text
    Args:
        request_id: Unique identifier for the request
        text: Text to convert to audio
        server_address: Address of the gRPC server
    Returns:
        Tuple of (success, s3_url, error_message)
    """
    client = AudioGRPCClient(server_address)
    try:
        return await client.generate_audio(request_id, text)
    finally:
        await client.disconnect()

if __name__ == '__main__':
    # Example async usage
    import uuid
    import asyncio
    async def main():
        request_id = str(uuid.uuid4())
        text = '''
Imagine standing on the Moon, where your feet sink into lifeless dust, shaped by countless meteor impacts. The craters tell stories of cosmic encounters, each a unique mark on this ancient landscape. Here, vast dark plains shimmer under the sun, contrasting with bright highlands that have weathered time. While Earth bursts with vibrant life, the Moon remains a captivating silence, quietly holding its history. Next time you gaze up at that glowing orb in the night sky, remember the mysteries it embodies and the bond it shares with our own world.
        '''


        success, s3_url, error = await generate_audio_simple(request_id, text)
        if success:
            print(f"✅ Success! Audio URL: {s3_url}")
        else:
            print(f"❌ Failed: {error}")
    asyncio.run(main()) 