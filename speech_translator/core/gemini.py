from google import genai
from google.genai import types
import logging
import os
from pathlib import Path
from typing import Optional
from speech_translator.config import Config

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        Config.validate()
        self.client = genai.Client(
            api_key=Config.GOOGLE_API_KEY,
            http_options={'api_version': Config.GEMINI_API_VERSION}
        )
        # Model for text understanding and translation (STT + Translation)
        self.thinking_model = Config.THINKING_MODEL
        # Model for speech generation (TTS)
        self.tts_model = Config.TTS_MODEL

    def list_models(self):
        """Lists available models from the Gemini API."""
        return self.client.models.list()

    def translate_audio(self, 
                       audio_file_path: str, 
                       target_lang: str, 
                       duration_hint_sec: Optional[float] = None,
                       voice_name: str = "Kore") -> bytes:
        """
        Two-step translation:
        1. Audio -> Translated Text (using configured thinking model)
        2. Translated Text -> Audio (using TTS model)
        """
        
        # --- Step 1: Get translation text ---
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        prompt_text = (
            f"Listen to this audio and translate the spoken content into {target_lang}. "
            "Output ONLY the translated text without any explanations, quotes, or timestamps."
        )

        logger.info(f"Step 1: Transcribing and translating to {target_lang}...")
        
        try:
            response_text = self.client.models.generate_content(
                model=self.thinking_model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
                            types.Part.from_text(text=prompt_text)
                        ]
                    )
                ]
            )
            
            if not response_text.text:
                logger.warning("No text translation generated.")
                raise ValueError("No text translation generated from Step 1.")
                
            translated_text = response_text.text.strip()
            logger.info(f"Translation result: {translated_text[:100]}...")

        except Exception as e:
            logger.error(f"Step 1 (Translation) failed: {e}")
            raise

        # --- Step 2: Generate audio (TTS) ---
        logger.info(f"Step 2: Generating speech from text using voice '{voice_name}'...")
        
        try:
            # Voice configuration (can be 'Puck', 'Kore', 'Fenrir', etc.)
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name 
                    )
                )
            )

            response_audio = self.client.models.generate_content(
                model=self.tts_model,
                contents=[
                    types.Content(
                        parts=[types.Part.from_text(text=translated_text)]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=speech_config
                )
            )

            # Extract audio data
            audio_data = None
            mime_type = None
            for part in response_audio.candidates[0].content.parts:
                if part.inline_data:
                    audio_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    break
            
            if not audio_data or len(audio_data) < 1000: # Less than 1kb is likely error or silence
                print(f"DEBUG: TTS returned invalid or too small audio data: {len(audio_data) if audio_data else 0} bytes", flush=True)
                raise ValueError("TTS returned invalid or empty audio data.")
            
            print(f"DEBUG: TTS generated {len(audio_data)} bytes of audio. MimeType: {mime_type}", flush=True)
            print(f"DEBUG: Audio Header (hex): {audio_data[:32].hex()}", flush=True)
            
            return audio_data

        except Exception as e:
            logger.error(f"Step 2 (TTS) failed: {e}")
            raise
