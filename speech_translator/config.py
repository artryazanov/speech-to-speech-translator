import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Defaults
    DEFAULT_MODEL_NAME = "gemini-2.5-flash"
    TEMP_DIR = Path("temp_audio")
    
    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set. Please check your .env file.")

# Ensure temp directory exists
Config.TEMP_DIR.mkdir(exist_ok=True)
