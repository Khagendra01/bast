from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import numpy as np
import random
import json
import threading
from typing import Dict, Optional

# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]

# Voice mapping - maps voice names to numbers for random selection
ENGLISH_VOICE_MAPPING = {
    1: "af_heart",
    2: "af_aoede", 
    3: "af_bella",
    4: "af_kore",
    5: "af_nicole",
    6: "af_sarah",
    7: "af_sky",
    8: "am_echo",
    9: "am_eric",
    10: "am_fenrir",
    11: "am_liam",
    12: "am_michael",
    13: "am_onyx",
    14: "am_puck",
}

SPANISH_VOICE_MAPPING = {
    1: "ef_dora",
    2: "em_alex",
    3: "em_santa",
}

# Language to voice mapping dictionary
LANGUAGE_VOICE_MAPPINGS = {
    'english': ENGLISH_VOICE_MAPPING,
    'spanish': SPANISH_VOICE_MAPPING,
    'a': ENGLISH_VOICE_MAPPING,  # American English
    'e': SPANISH_VOICE_MAPPING,  # Spanish
}

# Language code mapping
LANGUAGE_CODES = {
    'english': 'a',
    'spanish': 'e',
    'a': 'a',
    'e': 'e',
}


class PipelineManager:
    """Singleton class to manage pipeline instances with caching"""
    
    _instance = None
    _lock = threading.Lock()
    _pipelines: Dict[str, KPipeline] = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PipelineManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._pipelines = {}
            self._initialized = True
            print("PipelineManager initialized")
    
    def get_pipeline(self, language: str = 'english') -> KPipeline:
        """Get or create a pipeline for the specified language"""
        lang_code = LANGUAGE_CODES.get(language, 'a')
        
        if lang_code not in self._pipelines:
            with self._lock:
                if lang_code not in self._pipelines:
                    print(f"Creating new pipeline for language: {language} (code: {lang_code})")
                    self._pipelines[lang_code] = KPipeline(lang_code=lang_code, device='cuda')
                    print(f"Pipeline created successfully for {language}")
        
        return self._pipelines[lang_code]
    
    def clear_pipelines(self):
        """Clear all cached pipelines (useful for memory management)"""
        with self._lock:
            self._pipelines.clear()
            print("All pipelines cleared from cache")
    
    def get_cached_languages(self) -> list:
        """Get list of currently cached language codes"""
        return list(self._pipelines.keys())


# Global pipeline manager instance
_pipeline_manager = PipelineManager()


def get_random_voice(language='english'):
    """Generate a random number and return the corresponding voice for the given language"""
    voice_mapping = LANGUAGE_VOICE_MAPPINGS.get(language, ENGLISH_VOICE_MAPPING)
    random_number = random.randint(1, len(voice_mapping))
    selected_voice = voice_mapping[random_number]
    print(f"Language: {language}")
    print(f"Random number generated: {random_number}")
    print(f"Selected voice: {selected_voice}")
    return selected_voice


def create_pipeline(language='english'):
    """Create a pipeline with the appropriate language code (deprecated - use get_pipeline instead)"""
    return _pipeline_manager.get_pipeline(language)


def get_pipeline(language='english'):
    """Get a cached pipeline for the specified language"""
    return _pipeline_manager.get_pipeline(language)


def generate_audio(text, voice=None, language='english'):
    """Generate audio from text with the specified voice and language using cached pipeline"""
    try:
        # Use random voice if none provided
        if voice is None:
            voice = get_random_voice(language)
        
        # Get cached pipeline (no need to create new one)
        pipeline = get_pipeline(language)
        
        generator = pipeline(
            text, voice=voice,
            speed=1, split_pattern=r'\n+'
        )
        all_audio = []

        for i, (gs, ps, audio) in enumerate(generator):
            all_audio.append(audio)

        # Combine all audio segments into one
        if all_audio:
            combined_audio = np.concatenate(all_audio)
            sf.write('combined.wav', combined_audio, 24000)
            return combined_audio, 24000
        else:
            print("No audio generated")
            return None, None
            
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None, None


def generate_audio_with_pipeline(text, voice=None, language='english', pipeline=None):
    """Generate audio using a specific pipeline instance (for advanced usage)"""
    try:
        # Use random voice if none provided
        if voice is None:
            voice = get_random_voice(language)
        
        # Use provided pipeline or get cached one
        if pipeline is None:
            pipeline = get_pipeline(language)
        
        generator = pipeline(
            text, voice=voice,
            speed=1, split_pattern=r'\n+'
        )
        all_audio = []

        for i, (gs, ps, audio) in enumerate(generator):
            all_audio.append(audio)

        # Combine all audio segments into one
        if all_audio:
            combined_audio = np.concatenate(all_audio)
            return combined_audio, 24000
        else:
            print("No audio generated")
            return None, None
            
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None, None


def get_pipeline_manager():
    """Get the global pipeline manager instance"""
    return _pipeline_manager


if __name__ == "__main__":
    # Test the improved pipeline management
    print("Testing pipeline caching...")
    
    # Generate random voice and use it
    random_voice = get_random_voice('spanish')
    
    text = """Había un niño y una niña que estaban en un parque. Estaban hablando entre ellos sobre cuánto se amaban. El niño dijo: "Yo te amo más que nada". La niña respondió: "No, no siento lo mismo". El niño se enojó mucho. Salió a correr. Ahora ella lo está persiguiendo."""
    
    # First call - should create pipeline
    print("\n=== First call ===")
    generate_audio(text, voice=random_voice, language='spanish')
    
    # Second call - should use cached pipeline
    print("\n=== Second call (cached) ===")
    generate_audio(text, voice=random_voice, language='spanish')
    
    # Test English pipeline
    print("\n=== Testing English pipeline ===")
    generate_audio("Hello world, this is a test.", language='english')
    
    # Show cached languages
    print(f"\nCached languages: {get_pipeline_manager().get_cached_languages()}")








