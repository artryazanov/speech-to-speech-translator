from google import genai
from google.genai import types
import logging
import os
import json
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
        self.thinking_model = Config.THINKING_MODEL
        self.tts_model = Config.TTS_MODEL

    def list_models(self):
        """Lists available models from the Gemini API."""
        return self.client.models.list()

    def translate_audio(self, 
                       audio_file_path: str, 
                       target_lang: str, 
                       duration_hint_sec: Optional[float] = None,
                       voice_name: str = "Auto") -> bytes:
        """
        Two-step translation with Advanced Auto-Voice detection:
        1. Audio -> Translated Text + Detailed Speaker Classification
        2. Translated Text -> Audio (using mapped TTS voice)
        """
        
        # --- Step 1: Get translation and speaker info ---
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        logger.info(f"Step 1: Analyzing speaker and translating to {target_lang}...")

        translated_text = ""
        selected_voice = voice_name

        try:
            if voice_name == "Auto":
                # Extended prompt for age and gender detection
                prompt_text = (
                    f"Listen to this audio carefully.\n"
                    f"1. Analyze the speaker's voice to identify gender and approximate age.\n"
                    f"   Classify into one of these exact categories:\n"
                    f"   ['Boy', 'Young Man', 'Man', 'Elderly Man', 'Girl', 'Young Woman', 'Woman', 'Elderly Woman']\n"
                    f"2. Translate the spoken content into {target_lang}.\n"
                    f"3. Try to keep the speech duration close to {duration_hint_sec:.1f} seconds.\n\n"
                    f"Return ONLY a JSON object with this structure:\n"
                    f"{{\n"
                    f"  \"category\": \"CATEGORY_NAME\",\n"
                    f"  \"text\": \"TRANSLATED_TEXT\"\n"
                    f"}}"
                )
                
                response_text = self.client.models.generate_content(
                    model=self.thinking_model,
                    contents=[
                        types.Content(
                            parts=[
                                types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
                                types.Part.from_text(text=prompt_text)
                            ]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )

                if not response_text.text:
                    raise ValueError("No response from Gemini.")

                try:
                    data = json.loads(response_text.text)
                    translated_text = data.get("text", "").strip()
                    category = data.get("category", "Woman") # Fallback
                    
                    # Mapping logic: 8 categories -> 5 voices
                    # Voices: Puck (M), Charon (M-Deep), Fenrir (M-Strong), Kore (F), Aoede (F-High)
                    
                    voice_map = {
                        # Male categories
                        "Boy": "Puck",           # Young/high male
                        "Young Man": "Puck",     # Young male
                        "Man": "Fenrir",         # Adult confident
                        "Elderly Man": "Charon", # Deep/low for elderly
                        
                        # Female categories
                        "Girl": "Aoede",         # High/thin female
                        "Young Woman": "Aoede",  # Young female
                        "Woman": "Kore",         # Adult standard
                        "Elderly Woman": "Kore"  # (Kore sounds more mature than Aoede)
                    }
                    
                    selected_voice = voice_map.get(category, "Kore")
                        
                    logger.info(f"Detected: {category}. Selected voice: {selected_voice}")
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {response_text.text}")
                    translated_text = response_text.text
                    selected_voice = "Kore"

            else:
                # Manual voice selection (legacy behavior)
                prompt_text = (
                    f"Listen to this audio and translate the spoken content into {target_lang}. "
                    f"Try to keep the speech duration close to {duration_hint_sec:.1f} seconds. "
                    "Output ONLY the translated text without any explanations, quotes, or timestamps."
                )
                
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
                translated_text = response_text.text.strip()
                selected_voice = voice_name

            logger.info(f"Translation result: {translated_text[:100]}...")

        except Exception as e:
            logger.error(f"Step 1 (Translation/Analysis) failed: {e}")
            raise

        # --- Step 2: Generate audio (TTS) ---
        logger.info(f"Step 2: Generating speech using voice '{selected_voice}'...")
        
        try:
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=selected_voice 
                    )
                )
            )

            response_audio = self.client.models.generate_content(
                model=self.tts_model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_text(text=translated_text)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=speech_config
                )
            )

            audio_data = None
            for part in response_audio.candidates[0].content.parts:
                if part.inline_data:
                    audio_data = part.inline_data.data
                    break
            
            if not audio_data:
                raise ValueError("TTS returned invalid or empty audio data.")
            
            return audio_data

        except Exception as e:
            logger.error(f"Step 2 (TTS) failed: {e}")
            raise
