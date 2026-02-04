import google.generativeai as genai
import logging
from pathlib import Path
from typing import Optional
from speech_translator.config import Config

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        Config.validate()
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(Config.DEFAULT_MODEL_NAME)

    def upload_file(self, path: str):
        """Uploads a file to Gemini context."""
        logger.info(f"Uploading file {path} to Gemini...")
        try:
            sample_file = genai.upload_file(path=path)
            logger.info(f"File uploaded as: {sample_file.uri}")
            return sample_file
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

    def translate_audio(self, 
                       audio_file_path: str, 
                       target_lang: str, 
                       duration_hint_sec: Optional[float] = None) -> bytes:
        """
        Sends audio to Gemini and requests a speech-to-speech translation.
        Returns the raw audio bytes of the response.
        """
        
        # Upload the file first
        remote_file = self.upload_file(audio_file_path)
        
        prompt_parts = [
            f"Translate this speech to {target_lang}.",
            "INSTRUCTIONS:",
            "1. Output ONLY the translated audio file.",
            "2. PRESERVE the original voice, tone, emotion, and speaker identity exactly.",
            "3. Do not add background music or sound effects, only the voice.",
        ]
        
        if duration_hint_sec:
            prompt_parts.append(f"4. TIME ALIGNMENT: The output duration must be extremely close to {duration_hint_sec:.2f} seconds.")
            prompt_parts.append("5. Adjust speech rate naturally to fit this duration.")

        prompt = "\n".join(prompt_parts)
        
        logger.info(f"Requesting translation to {target_lang} with duration hint: {duration_hint_sec}")
        
        try:
            response = self.model.generate_content(
                [prompt, remote_file],
                generation_config={"response_mime_type": "audio/mpeg"}
            )
            
            # Check for parts
            if not response.candidates:
                raise ValueError("No candidates returned from Gemini.")
            
            part = response.candidates[0].content.parts[0]
            if not part.inline_data:
                raise ValueError("No inline audio data in response.")
                
            return part.inline_data.data
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
