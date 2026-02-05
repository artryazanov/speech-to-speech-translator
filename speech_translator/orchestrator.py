import logging
import os
from pydub import AudioSegment
from pathlib import Path
from speech_translator.config import Config
from speech_translator.core.audio import AudioProcessor
from speech_translator.core.gemini import GeminiClient
from speech_translator.core.downloader import download_content

logger = logging.getLogger(__name__)

class TranslationOrchestrator:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.gemini_client = GeminiClient()

    def process(self, input_path: str, output_path: str, target_lang: str, ducking: bool = False):
        """
        Main pipeline: Load -> Split -> Translate -> Merge -> (Broadcast/Duck) -> Save.
        """
        # Determine if input is URL or File
        is_url = input_path.startswith("http://") or input_path.startswith("https://")
        downloaded_temp_file = None

        if is_url:
            logger.info("URL detected. Initiating download...")
            try:
                downloaded_temp_file = download_content(input_path, Config.TEMP_DIR)
                input_file = downloaded_temp_file
            except Exception as e:
                raise RuntimeError(f"Could not download content from URL: {e}")
        else:
            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Starting processing for {input_file}")
        
        try:
            # 1. Load Audio
            # Pydub automatically handles video files by extracting the audio track (using ffmpeg)
            if input_file.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
                logger.info(f"Video file detected: {input_file.name}. Extracting audio track...")
                
            original_audio = self.audio_processor.load_audio(str(input_file))
            total_duration = len(original_audio) / 1000.0
            
            # 2. Strategy Decision: Split or One-shot?
            # Gemini Flash 2.5 context is large, but for reliable timestamps and shorter failures, 
            # chunks of ~30-60s are safer.
            chunks = self.audio_processor.split_on_silence_smart(original_audio, target_chunk_len_sec=45)
            
            translated_segments = []
            
            for i, chunk in enumerate(chunks):
                chunk_duration = len(chunk) / 1000.0
                logger.info(f"Processing chunk {i+1}/{len(chunks)} ({chunk_duration:.2f}s)...")
                
                # Save temp chunk
                temp_chunk_path = Config.TEMP_DIR / f"temp_chunk_{i}.mp3"
                self.audio_processor.save_audio(chunk, str(temp_chunk_path))
                
                import time
                max_retries = 3
                retry_delay = 60
                
                translated_chunk_success = False
                
                for attempt in range(max_retries):
                    try:
                        # API Call
                        # We ask Gemini to match the duration of the chunk for sync
                        logger.info(f"Sending audio chunk to Gemini. Size: {len(chunk)} bytes ({chunk_duration:.2f}s) - Attempt {attempt+1}")
                        
                        translated_bytes = self.gemini_client.translate_audio(
                            str(temp_chunk_path), 
                            target_lang, 
                            duration_hint_sec=chunk_duration
                        )
                        
                        # Write response to temp file to load back into pydub
                        temp_out_path = Config.TEMP_DIR / f"translated_chunk_{i}.mp3"
                        with open(temp_out_path, "wb") as f:
                            f.write(translated_bytes)
                        
                        translated_segment = self.audio_processor.load_audio(str(temp_out_path))
                        
                        # Check Duration & Adjust if strictly needed (simple fallback)
                        actual_duration = len(translated_segment) / 1000.0
                        diff = abs(actual_duration - chunk_duration)
                        if diff > 1.0: # If more than 1 second off
                            logger.warning(f"Drift detected in chunk {i}: target {chunk_duration}s, got {actual_duration}s. Attempting speed correction.")
                            translated_segment = self.audio_processor.speed_match(translated_segment, chunk_duration)
                        
                        translated_segments.append(translated_segment)
                        translated_chunk_success = True
                        break # Success, exit retry loop
                        
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            logger.warning(f"Rate limit hit for chunk {i}: {e}")
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying in {retry_delay}s...")
                                time.sleep(retry_delay)
                                continue
                        
                        # If not rate limit or retries exhausted, re-raise to handle as failure
                        logger.error(f"Failed to process chunk {i} on attempt {attempt+1}: {e}")
                        if attempt == max_retries - 1:
                            # Final failure
                            logger.error("Inserting silence for failed chunk to maintain sync.")
                            translated_segments.append(AudioSegment.silent(duration=len(chunk)))
                
                # Cleanup temp files
                if temp_chunk_path.exists(): os.remove(temp_chunk_path)
                temp_translated_path = Config.TEMP_DIR / f"translated_chunk_{i}.mp3"
                if temp_translated_path.exists(): os.remove(temp_translated_path)

                # Rate Limiting for Experimental Models (Preventative)
                if i < len(chunks) - 1 and translated_chunk_success:
                    logger.info("Sleeping 40s to respect rate limits (2 RPM)...")
                    time.sleep(40)

            # 3. Merge
            logger.info("Merging processed segments...")
            final_voice = self.audio_processor.merge_segments(translated_segments)
            
            # 4. Post-processing (Ducking)
            if ducking:
                logger.info("Applying audio ducking...")
                # We need the original full audio again for the base
                # Ducking requires the original track (background) + new voice
                final_output = self.audio_processor.apply_ducking(original_audio, final_voice)
            else:
                final_output = final_voice

            # 5. Export
            logger.info(f"Saving final output to {output_path}")
            self.audio_processor.save_audio(final_output, output_path)
            logger.info("Done!")
            
        finally:
            if downloaded_temp_file and downloaded_temp_file.exists():
                logger.info(f"Cleaning up downloaded file: {downloaded_temp_file}")
                os.remove(downloaded_temp_file)
