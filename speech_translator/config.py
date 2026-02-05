import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Defaults with env overrides
    THINKING_MODEL = os.getenv("THINKING_MODEL", "gemini-3-pro-preview")
    TTS_MODEL = os.getenv("TTS_MODEL", "gemini-2.5-pro-preview-tts")
    GEMINI_API_VERSION = os.getenv("GEMINI_API_VERSION", "v1beta")
    TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp_audio"))
    
    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please check your .env file.")

# Ensure temp directory exists
Config.TEMP_DIR.mkdir(exist_ok=True)
