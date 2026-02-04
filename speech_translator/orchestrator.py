import logging
import os
from pathlib import Path
from speech_translator.config import Config
from speech_translator.core.audio import AudioProcessor
from speech_translator.core.gemini import GeminiClient

logger = logging.getLogger(__name__)

class TranslationOrchestrator:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.gemini_client = GeminiClient()

    def process(self, input_path: str, output_path: str, target_lang: str, ducking: bool = False):
        """
        Main pipeline: Load -> Split -> Translate -> Merge -> (Broadcast/Duck) -> Save.
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Starting processing for {input_path}")
        
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
            
            try:
                # API Call
                # We ask Gemini to match the duration of the chunk for sync
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
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i}: {e}. Skipping/Silence insertion?")
                # Fallback: Insert silence or original audio? 
                # For now, let's allow failure to bubble up or insert silence.
                # Inserting silence to keep sync
                logger.error("Inserting silence for failed chunk to maintain sync.")
                translated_segments.append(AudioSegment.silent(duration=len(chunk)))
            finally:
                # Cleanup temp files
                if temp_chunk_path.exists(): os.remove(temp_chunk_path)
                # Translated temp file cleans up automatically? No, manual clean
                # We reuse the path name, so it will be overwritten or we can clean later.
                # Let's clean immediate to avoid disk clutter
                temp_translated_path = Config.TEMP_DIR / f"translated_chunk_{i}.mp3"
                if temp_translated_path.exists(): os.remove(temp_translated_path)

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
