import grpc
import logging
import os
import uuid
import re
from concurrent import futures
import boto3
from botocore.exceptions import ClientError
import torch
import numpy as np
import io
import soundfile as sf
from TTS.api import TTS
from dotenv import load_dotenv
from bson.binary import UuidRepresentation
import bg_gen.Ambient_Pipeline as ap
import loco
# Import generated protobuf classes
import audio_service_pb2
import audio_service_pb2_grpc

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_FOLDER_PATH = os.getenv('S3_FOLDER_PATH', 'audio-files')

# Validate AWS credentials
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
    logger.warning("AWS credentials not found in environment variables. S3 upload will be skipped.")
    logger.info("Please create a .env file with the following variables:")
    logger.info("AWS_ACCESS_KEY_ID=your_access_key")
    logger.info("AWS_SECRET_ACCESS_KEY=your_secret_key")
    logger.info("S3_BUCKET_NAME=your-bucket-name")
    logger.info("AWS_DEFAULT_REGION=us-east-1 (optional)")
    logger.info("S3_FOLDER_PATH=audio-files (optional)")

class AudioServiceServicer(audio_service_pb2_grpc.AudioServiceServicer):
    def __init__(self):
        # Initialize TTS model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load Tacotron2-DDC model
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(self.device)
        
        # Initialize S3 client if credentials are available
        self.s3_client = None
        if all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_DEFAULT_REGION
                )
                logger.info(f"✅ S3 client initialized for region: {AWS_DEFAULT_REGION}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize S3 client: {e}")
                self.s3_client = None
        
        logger.info("AudioService initialized successfully")

    def _clean_text(self, text: str) -> str:
        """
        Clean text for TTS processing following Coqui TTS best practices.
        
        Coqui TTS works best with clean, plain text without special formatting.
        This function removes markdown, special characters, and normalizes text.
        """
        import re
        
        # Step 1: Remove markdown formatting
        # Remove headers (# ## ### etc.)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove bold (**text** or __text__)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        
        # Remove italic (*text* or _text_)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        
        # Remove code blocks (```code```)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove inline code (`code`)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove table formatting (| text |)
        text = re.sub(r'\|', ' ', text)
        
        # Remove horizontal rules (--- or ***)
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
        
        # Step 2: Remove problematic characters that can cause TTS issues
        # Remove special characters that might confuse the TTS model
        text = re.sub(r'[#@$%^&*+=<>[\]{}|\\~`]', ' ', text)
        
        # Step 3: Clean up whitespace and formatting
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Step 4: Handle empty or very short text
        if len(text) < 10:
            text = "Text to speech conversion."
        elif len(text) < 20:
            text = "Text to speech conversion. " + text
        
        # Step 5: Ensure proper sentence endings
        # Add period if text doesn't end with punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        # Step 6: Normalize quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('…', '...')
        
        # Step 7: Remove any remaining control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text

    def _generate_audio_fast(self, text: str) -> tuple:
        """
        Generate audio with robust error handling for different TTS model outputs.
        
        Coqui TTS models can return different output formats:
        - Some return just the waveform
        - Some return (waveform, sample_rate)
        - Some return (waveform, sample_rate, alignment)
        """
        try:
            # Generate audio using TTS model
            output = self.tts.tts(text)
            
            # Handle different return types from TTS models
            if isinstance(output, tuple):
                if len(output) == 2:
                    wav, sample_rate = output
                elif len(output) == 3:
                    wav, sample_rate, _ = output  # Ignore alignment
                else:
                    # Fallback: assume first element is waveform
                    wav = output[0]
                    sample_rate = 22050
            else:
                # Single output - assume it's the waveform
                wav = output
                sample_rate = 22050
            
            # Ensure wav is numpy array
            if not hasattr(wav, 'shape'):
                wav = np.array(wav)
            
            # Normalize audio if needed
            if wav.max() > 1.0 or wav.min() < -1.0:
                wav = wav / max(abs(wav.max()), abs(wav.min()))
            
            return wav, sample_rate
            
        except Exception as e:
            logger.warning(f"Primary TTS generation failed: {e}")
            
            # Fallback: try with simplified text
            try:
                simplified_text = re.sub(r'[^\w\s.,!?]', ' ', text).strip()
                if len(simplified_text) < 10:
                    simplified_text = "Text to speech conversion."
                
                output = self.tts.tts(simplified_text)
                
                # Handle different return types
                if isinstance(output, tuple):
                    if len(output) == 2:
                        wav, sample_rate = output
                    elif len(output) == 3:
                        wav, sample_rate, _ = output
                    else:
                        wav = output[0]
                        sample_rate = 22050
                else:
                    wav = output
                    sample_rate = 22050
                
                # Ensure wav is numpy array
                if not hasattr(wav, 'shape'):
                    wav = np.array(wav)
                
                # Normalize audio
                if wav.max() > 1.0 or wav.min() < -1.0:
                    wav = wav / max(abs(wav.max()), abs(wav.min()))
                
                return wav, sample_rate
                
            except Exception as fallback_error:
                logger.error(f"Fallback TTS generation also failed: {fallback_error}")
                
                # Final fallback: generate simple sine wave
                sample_rate = 22050
                duration = 2.0
                frequency = 440  # A4 note
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                wav = np.sin(2 * np.pi * frequency * t) * 0.3
                
                return wav, sample_rate

    def _upload_audio_blob_to_s3(self, audio_blob: bytes, s3_key: str, content_type: str = 'audio/wav') -> bool:
        """Upload audio blob directly to S3 bucket"""
        if not self.s3_client:
            logger.warning("S3 client not available. Skipping upload.")
            return False
        
        try:
            self.s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=audio_blob,
                ContentType=content_type
            )
            logger.info(f"✅ Successfully uploaded audio blob to S3: s3://{S3_BUCKET_NAME}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to upload audio blob to S3: {e}")
            return False

    def GenerateAudio(self, request, context):
        """Generate audio from text and upload to S3"""
        try:
            language = request.id
            text = request.text
            
            logger.info(f"Generating audio for ID: {language}")
            
            # Clean and prepare text
            try:
                cleaned_text = self._clean_text(text)
                logger.info(f"Text cleaned successfully, length: {len(cleaned_text)}")
            except Exception as e:
                logger.error(f"Text cleaning failed: {e}")
                raise
            
            # Voice generation
            try:
                #wav, sample_rate = self._generate_audio_fast(cleaned_text)
                # Use the improved pipeline caching - no need to pass request_id as voice
                wav, sample_rate = loco.generate_audio(cleaned_text, language=language)
                logger.info(f"Voice generated: shape={wav.shape}, dtype={wav.dtype}, sr={sample_rate}")
                logger.info(f"Voice duration: {len(wav) / sample_rate:.2f}s")
                
                # Validate and convert voice output to numpy if needed
                if hasattr(wav, 'cpu'):  # PyTorch tensor
                    wav = wav.cpu().numpy()
                wav = np.array(wav, dtype=np.float32)
                if len(wav.shape) > 1:
                    wav = wav.squeeze()  # Remove extra dimensions
                    
                if len(wav.shape) != 1:
                    raise ValueError(f"Invalid wav shape after processing: {wav.shape}")
                if sample_rate <= 0:
                    raise ValueError(f"Invalid sample rate: {sample_rate}")
                    
            except Exception as e:
                logger.error(f"Voice generation failed: {e}")
                raise
            
            # Music generation
            try:
                music, music_sr = ap.text_to_music(cleaned_text)
                #music, music_sr = np.zeros(100000), 24000
                logger.info(f"Music generated: shape={music.shape}, dtype={music.dtype}, sr={music_sr}")
                logger.info(f"Music duration: {len(music) / music_sr:.2f}s")
                
                # Convert PyTorch tensor to numpy and handle dimensions
                if hasattr(music, 'cpu'):  # PyTorch tensor
                    music = music.cpu().numpy()
                
                music = np.array(music, dtype=np.float32)
                
                # Handle different tensor shapes
                if len(music.shape) == 3:  # [batch, channels, samples] -> [samples]
                    music = music.squeeze()  # Remove batch and channel dims
                elif len(music.shape) == 2:  # [channels, samples] or [batch, samples]
                    music = music.squeeze()  # Remove extra dim
                
                # Ensure we have 1D array
                if len(music.shape) != 1:
                    raise ValueError(f"Invalid music shape after processing: {music.shape}")
                if music_sr <= 0:
                    raise ValueError(f"Invalid music sample rate: {music_sr}")
                
                logger.info(f"Music processed: final shape={music.shape}, dtype={music.dtype}")
                    
            except Exception as e:
                logger.error(f"Music generation failed: {e}")
                raise
            

            # --- Combine voice and music ---
            try:
                # Resample music if needed
                if music_sr != sample_rate:
                    try:
                        import librosa
                        logger.info(f"Resampling music from {music_sr} to {sample_rate}")
                        original_music_length = len(music)
                        music = librosa.resample(music, orig_sr=music_sr, target_sr=sample_rate)
                        logger.info(f"Music resampled: {original_music_length} -> {len(music)} samples")
                    except ImportError:
                        logger.error("librosa not available for resampling")
                        raise
                    except Exception as e:
                        logger.error(f"Resampling failed: {e}")
                        raise
                
                logger.info(f"Before combining - wav: {wav.shape}, music: {music.shape}")
                
                # Pad or trim music to match voice length
                if len(music) < len(wav):
                    music = np.pad(music, (0, len(wav) - len(music)), mode='constant')
                    logger.info(f"Music padded to {music.shape}")
                elif len(music) > len(wav):
                    music = music[:len(wav)]
                    logger.info(f"Music trimmed to {music.shape}")
                
                # Ensure both arrays are float32 and same shape
                wav = wav.astype(np.float32)
                music = music.astype(np.float32)
                
                # Mix: voice at full volume, music at 30%
                combined = wav + 0.2 * music
                logger.info(f"Audio combined: shape={combined.shape}, dtype={combined.dtype}")
                
                # Clip to [-1, 1] to avoid overflow
                combined = np.clip(combined, -1.0, 1.0)
                
            except Exception as e:
                logger.error(f"Audio combination failed: {e}")
                raise

            # Encode combined waveform as WAV into an in-memory buffer
            try:
                buffer = io.BytesIO()
                sf.write(buffer, combined, sample_rate, format="WAV")
                audio_blob = buffer.getvalue()
                logger.info(f"Audio encoded to WAV, size: {len(audio_blob)} bytes")
                
            except Exception as e:
                logger.error(f"WAV encoding failed: {e}")
                raise
            
            # Upload to S3 if credentials are available
            try:
                s3_url = ""
                language = language + "_" + str(uuid.uuid4())
                if self.s3_client:
                    # Create S3 key with folder structure
                    s3_key = f"{S3_FOLDER_PATH}/{language}.wav"
                    
                    # Upload the audio blob
                    self._upload_audio_blob_to_s3(audio_blob, s3_key)
                    s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
                    logger.info(f"Audio uploaded to S3: {s3_url}")
                else:
                    logger.info("No S3 client, skipping upload")
                
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")
                # Don't raise here, continue with response
            
            # Calculate duration in seconds
            duration_sec = len(combined) / sample_rate
            duration_str = str(int(round(duration_sec)))
            
            # Add detailed logging for debugging
            logger.info(f"Final audio stats - Length: {len(combined)}, Sample Rate: {sample_rate}, Duration: {duration_sec:.2f}s, Rounded: {duration_str}s")
            
            return audio_service_pb2.GenerateAudioResponse(
                id=language,
                s3_url=s3_url,
                success=True,
                error_message=duration_str
            )
            
        except Exception as e:
            error_msg = f"Error generating audio: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Quick fallback - create simple audio
            try:
                sample_rate = 22050
                duration = 1.0
                frequency = 440
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                wav = np.sin(2 * np.pi * frequency * t) * 0.3
                
                # Encode as WAV
                buffer = io.BytesIO()
                sf.write(buffer, wav, sample_rate, format="WAV")
                audio_blob = buffer.getvalue()
                
                # Upload fallback audio to S3
                s3_url = ""
                if self.s3_client:
                    s3_key = f"{S3_FOLDER_PATH}/{language}_fallback.wav"
                    self._upload_audio_blob_to_s3(audio_blob, s3_key)
                    s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
                
                return audio_service_pb2.GenerateAudioResponse(
                    id=language,
                    s3_url=s3_url,
                    success=True,
                    error_message=f"Generated fallback audio due to: {error_msg}"
                )
                
            except Exception as fallback_e:
                logger.error(f"Fallback generation also failed: {fallback_e}")
                return audio_service_pb2.GenerateAudioResponse(
                    id=language,
                    s3_url="",
                    success=False,
                    error_message=f"Complete failure: {error_msg}"
                )

def serve():
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    audio_service_pb2_grpc.add_AudioServiceServicer_to_server(
        AudioServiceServicer(), server
    )
    
    # Get port from environment or use default
    port = int(os.getenv('GRPC_PORT', 50051))
    server.add_insecure_port(f'[::]:{port}')
    
    logger.info(f"Starting gRPC server on port {port}")
    server.start()
    logger.info("gRPC server started successfully")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server")
        server.stop(0)

if __name__ == '__main__':
    serve() 