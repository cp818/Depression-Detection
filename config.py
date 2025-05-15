import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Deepgram API settings
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    
    # Audio settings
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_CHUNK_SIZE: int = 4096
    
    # Depression detection settings
    DEPRESSION_SCORE_THRESHOLD_LOW: float = 20.0
    DEPRESSION_SCORE_THRESHOLD_MILD: float = 40.0
    DEPRESSION_SCORE_THRESHOLD_MODERATE: float = 60.0
    DEPRESSION_SCORE_THRESHOLD_HIGH: float = 80.0
    
    # Feature weights in depression scoring
    WEIGHT_NEGATIVE_SENTIMENT: float = 2.5
    WEIGHT_DEPRESSION_KEYWORDS: float = 2.0
    WEIGHT_FIRST_PERSON_FOCUS: float = 1.0
    WEIGHT_SPEECH_RATE: float = 1.5
    WEIGHT_WORD_VARIETY: float = 1.0
    WEIGHT_PAUSE_FREQUENCY: float = 1.0
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Storage settings
    SAVE_RECORDINGS: bool = False
    RECORDINGS_DIR: str = "recordings"
    
    # Model settings
    DEEPGRAM_MODEL: str = "nova-3"
    DEEPGRAM_LANGUAGE: str = "en"
    DEEPGRAM_TIER: str = "enhanced"
    
    class Config:
        env_file = ".env"
        
def get_settings() -> AppSettings:
    """Get application settings instance."""
    return AppSettings()

# Global settings instance
settings = get_settings()

# Create recordings directory if needed
if settings.SAVE_RECORDINGS and not os.path.exists(settings.RECORDINGS_DIR):
    os.makedirs(settings.RECORDINGS_DIR, exist_ok=True)
