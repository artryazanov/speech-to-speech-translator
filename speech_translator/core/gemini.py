from google import genai
from google.genai import types
import logging
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
        self.model_name = Config.DEFAULT_MODEL_NAME

    def translate_audio(self, 
                       audio_file_path: str, 
                       target_lang: str, 
                       duration_hint_sec: Optional[float] = None) -> bytes:
        """
        Sends audio to Gemini and requests a speech-to-speech translation.
        Returns the raw audio bytes of the response.
        """
        
        # Read file content directly as bytes usually works better for simple scripts 
        # than managing uploads unless files are huge. But for S2S, uploads are safer.
        # The new SDK handles local file paths seamlessly in generate_content for some cases,
        # but let's stick to the official pattern or simple bytes if supported.
        # Actually, new SDK allows passing file path directly if using the File API equivalent,
        # or we can just read bytes.
        
        # Let's read the file as bytes for simplicity and reliability with the new SDK
        # if the file is small enough (chunks are small).
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
            
        prompt_parts = [
            f"Translate this speech to {target_lang}.",
            "INSTRUCTIONS:",
            "1. Output ONLY the translated audio.",
            "2. PRESERVE the original voice, tone, emotion, and speaker identity exactly.",
            "3. Do not add background music or sound effects, only the voice.",
        ]
        
        if duration_hint_sec:
            prompt_parts.append(f"4. TIME ALIGNMENT: The output duration must be extremely close to {duration_hint_sec:.2f} seconds.")
            prompt_parts.append("5. Adjust speech rate naturally to fit this duration.")

        prompt = "\n".join(prompt_parts)
        
        logger.info(f"Requesting translation to {target_lang} with duration hint: {duration_hint_sec}")
        
        try:
            # New SDK call structure
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"]
                )
            )
            
            # Extract audio from response
            if not response.candidates:
                logger.error("No candidates returned from Gemini.")
                logger.error(f"Response feedback: {response.prompt_feedback}")
                raise ValueError("No candidates returned from Gemini.")
                
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
            
            # Log the text content if audio is missing to understand why
            text_content = ""
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_content += part.text
            
            logger.error("No inline audio data found in response.")
            logger.error(f"Finish Reason: {response.candidates[0].finish_reason}")
            logger.error(f"Safety Ratings: {response.candidates[0].safety_ratings}")
            logger.error(f"Text Content (if any): {text_content}")
            
            raise ValueError(f"No inline audio data found. Model said: {text_content[:200]}...")
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
