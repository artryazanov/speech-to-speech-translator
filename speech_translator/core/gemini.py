from google import genai
from google.genai import types
import logging
import os
import json
import io
import time
import re
from pydub import AudioSegment
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
        # Rate Limiting: 10 RPM
        self._tts_request_times = [] 
        self._rpm_limit = 10
        self._rpm_window = 60.0

    def list_models(self):
        """Lists available models from the Gemini API."""
        return self.client.models.list()

    def _get_voice_for_category(self, category: str, speaker_id: str = "A") -> str:
        """Helper to map speaker characteristics to Gemini Voices."""
        category = category.lower()
        if "boy" in category or "young man" in category:
            return "Puck"
        elif "elderly man" in category or "deep" in category:
            return "Charon"
        elif "girl" in category or "young woman" in category:
            return "Aoede"
        elif "elderly woman" in category:
             return "Kore"
        elif "woman" in category or "female" in category:
            return "Kore"
        elif "man" in category or "male" in category:
            return "Fenrir"
        elif "elderly" in category: # Generic Elderly
            return "Kore"
        else: # Default
            return "Kore"

    def translate_audio(self, 
                       audio_file_path: str, 
                       target_lang: str, 
                       duration_hint_sec: Optional[float] = None,
                       voice_name: str = "Auto",
                       mode: str = "monologue") -> bytes:
        """
        Handles both Monologue and Dialogue modes.
        Returns: raw audio bytes (MP3/WAV).
        """
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        logger.info(f"Analyzing audio (Mode: {mode}) for translation to {target_lang}...")

        if mode == "monologue":
            return self._process_monologue(audio_bytes, target_lang, duration_hint_sec, voice_name)
        else:
            return self._process_dialogue(audio_bytes, target_lang, duration_hint_sec)

    def _process_monologue(self, audio_bytes: bytes, target_lang: str, duration_hint_sec: Optional[float], voice_name: str) -> bytes:
        """
        Two-step translation with Advanced Auto-Voice detection (Legacy/Monologue Mode):
        1. Audio -> Translated Text + Detailed Speaker Classification
        2. Translated Text -> Audio (using mapped TTS voice)
        """
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
                        
                    logger.info(f"--- Detected Speaker (Monologue) ---")
                    logger.info(f"Category: {category} -> Voice: {selected_voice}")
                    logger.info(f"------------------------------------")
                    
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
        return self._generate_tts(translated_text, selected_voice)

    def _process_dialogue(self, audio_bytes: bytes, target_lang: str, duration_hint_sec: Optional[float]) -> bytes:
        """
        Advanced handling for multiple speakers.
        1. Diarize & Translate -> List of segments.
        2. TTS each segment with specific voice.
        3. Stitch audio together.
        """
        # 1. Prompt for structured dialogue
        prompt_text = (
            f"You are a professional dubbing director. Listen to this audio chunk.\n"
            f"It contains a dialogue or multiple speakers.\n"
            f"1. Identify distinct phrases/turns.\n"
            f"2. For each phrase, identify the Speaker (A, B, C...).\n"
            f"   IMPORTANT: Be conservative with speaker splitting. Only assign a new Speaker ID if you are certain it is a different person.\n"
            f"   Do not split speakers based on intonation, emotion, or short pauses. If the voice sounds similar, treat it as the same speaker.\n"
            f"3. Classify each speaker's voice: ['Boy', 'Man', 'Deep Man', 'Girl', 'Woman', 'Elderly'].\n"
            f"4. Translate the phrase to {target_lang}.\n"
            f"{f'5. Try to keep the total speech duration close to {duration_hint_sec:.1f} seconds.' if duration_hint_sec else ''}\n"
            f"Return ONLY a JSON object with a list 'segments':\n"
            f"{{\n"
            f"  \"segments\": [\n"
            f"    {{\"speaker\": \"A\", \"category\": \"Man\", \"text\": \"Hello there.\", \"approx_duration_ratio\": 0.2}},\n"
            f"    {{\"speaker\": \"B\", \"category\": \"Woman\", \"text\": \"Hi! How are you?\", \"approx_duration_ratio\": 0.8}}\n"
            f"  ]\n"
            f"}}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.thinking_model,
                contents=[
                    types.Content(parts=[
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
                        types.Part.from_text(text=prompt_text)
                    ])
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            
            if not response.text:
                raise ValueError("Empty response from Gemini Dialogue processing.")
                
            data = json.loads(response.text)
            segments = data.get("segments", [])
            
            if not segments:
                logger.warning("No segments detected in dialogue. Falling back to simple translation.")
                # Fallback: treat everything as one text
                full_text = " ".join([s.get("text", "") for s in segments])
                return self._generate_tts(full_text, "Kore")

            # 2. Process segments and generate Audio
            combined_audio = AudioSegment.empty()
            
            # Print Speaker Summary
            unique_speakers = {}
            for s in segments:
                spk_id = s.get("speaker", "?")
                cat = s.get("category", "?")
                if spk_id not in unique_speakers:
                     unique_speakers[spk_id] = {
                         "category": cat,
                         "voice": self._get_voice_for_category(cat)
                     }
            
            logger.info(f"--- Detected Speakers in Chunk ---")
            for spk_id, info in sorted(unique_speakers.items()):
                logger.info(f"Speaker {spk_id}: {info['category']} -> {info['voice']}")
            logger.info(f"----------------------------------")

            logger.info(f"Detected {len(segments)} segments in dialogue.")
            
            for i, seg in enumerate(segments):
                text = seg.get("text", "").strip()
                category = seg.get("category", "Woman")
                speaker_id = seg.get("speaker", "A")
                
                # Clean text for TTS stability
                text = self._clean_text_for_tts(text)

                if not text:
                    continue
                    
                # Choose voice
                voice = self._get_voice_for_category(category, speaker_id)
                logger.info(f"  [{i+1}] Spk {speaker_id} ({category}) -> {voice}: '{text[:30]}...'")
                
                # Generate audio for segment
                audio_bytes_chunk = self._generate_tts(text, voice)
                
                # Convert bytes -> AudioSegment
                try:
                    seg_audio = self._load_audio_bytes(audio_bytes_chunk)
                    combined_audio += seg_audio
                    # Optional: Add small pause between segments if needed
                except Exception as e:
                    logger.error(f"Error processing segment audio: {e}")
            
            # 3. Return combined bytes
            output_buffer = io.BytesIO()
            combined_audio.export(output_buffer, format="mp3")
            return output_buffer.getvalue()

        except Exception as e:
            logger.error(f"Dialogue processing failed: {e}")
            raise

    def _load_audio_bytes(self, audio_data: bytes) -> AudioSegment:
        """Helper to load audio bytes that might be Raw PCM (Gemini default) or MP3/WAV."""
        if audio_data.startswith(b'RIFF'):
            return AudioSegment.from_wav(io.BytesIO(audio_data))
        elif audio_data.startswith(b'ID3') or audio_data.startswith(b'\xFF\xFB'):
            return AudioSegment.from_file(io.BytesIO(audio_data))
        else:
             # Assume Raw PCM 24kHz 16bit Mono (Gemini standard)
            return AudioSegment(
                data=audio_data,
                sample_width=2,
                frame_rate=24000,
                channels=1
            )

    def _wait_for_rate_limit(self):
        """Enforces 10 RPM limit for TTS requests."""
        current_time = time.time()
        # Remove timestamps older than 60 seconds
        self._tts_request_times = [t for t in self._tts_request_times if current_time - t < self._rpm_window]
        
        if len(self._tts_request_times) >= self._rpm_limit:
            # Wait until the oldest request expires
            oldest_request = self._tts_request_times[0]
            wait_time = self._rpm_window - (current_time - oldest_request) + 0.1 # Buffer
            if wait_time > 0:
                logger.info(f"Rate limit reached ({len(self._tts_request_times)}/{self._rpm_limit}). Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
            
            # Re-clean after wait
            current_time = time.time()
            self._tts_request_times = [t for t in self._tts_request_times if current_time - t < self._rpm_window]

        self._tts_request_times.append(current_time)

    def _clean_text_for_tts(self, text: str) -> str:
        """
        Cleans text to prevent TTS errors (e.g., empty content, excessive punctuation).
        Returns cleaned text or empty string if invalid.
        """
        if not text:
            return ""
        
        # 1. Replace multiple dots/ellipses with single dot (e.g. "Yeah...." -> "Yeah.")
        # This helps because Gemini TTS sometimes chokes on long ellipses.
        text = re.sub(r'\.{2,}', '.', text)
        
        # 2. whitespace cleanup
        text = text.strip()
        
        # 3. Check if text is only punctuation
        if not any(c.isalnum() for c in text):
            return ""
            
        return text

    def _generate_tts(self, text: str, voice_name: str) -> bytes:
        """Internal helper to call TTS model for a piece of text."""
        logger.info(f"Generating speech using voice '{voice_name}'...")
        
        max_retries = 3
        retry_delay = 30 # Default if no retry-after header
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                speech_config = types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                    )
                )
                response = self.client.models.generate_content(
                    model=self.tts_model,
                    contents=[types.Content(parts=[types.Part.from_text(text=text)])],
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=speech_config
                    )
                )
                
                if not response.candidates:
                    raise ValueError("TTS API returned no candidates.")
                
                candidate = response.candidates[0]
                if not candidate.content:
                     # Check for safety blocks or other reasons
                    finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
                    safety_ratings = getattr(candidate, 'safety_ratings', [])
                    raise ValueError(f"TTS API returned no content. Finish reason: {finish_reason}. Safety: {safety_ratings}")
                
                for part in candidate.content.parts:
                    if part.inline_data:
                        return part.inline_data.data
                raise ValueError("No audio data in TTS response")

            except Exception as e:
                is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_rate_limit:
                    logger.warning(f"TTS Rate limit hit for voice {voice_name}: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying TTS in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                
                logger.error(f"TTS gen failed for voice {voice_name} (Attempt {attempt+1}): {e}")
                logger.error(f"Failed Text: '{text[:100]}...'")
                
                # Add a small delay even for non-rate-limit errors to avoid rapid-fire failures
                time.sleep(2)
                
                if attempt == max_retries - 1:
                    # Fallback to Kore voice if possible and not already using it
                    if voice_name != "Kore":
                         logger.warning(f"Failed with voice {voice_name}. Retrying one last time with fallback voice 'Kore'...")
                         return self._generate_tts(text, "Kore")
                    raise

        raise ValueError("TTS generation failed unexpectedly")
