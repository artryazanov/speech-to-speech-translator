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

    def process(self, input_path: str, output_path: str, target_lang: str, ducking: bool = False, voice_name: str = "Kore", mode: str = "monologue"):
        """
        Main pipeline: Load -> Split -> Translate -> Merge -> (Broadcast/Duck) -> Save.
        """
        # Determine output format and need for video early
        is_video_output = Path(output_path).suffix.lower() in [".mp4", ".mov", ".mkv", ".avi", ".webm"]
        
        # Determine if input is URL or File
        is_url = input_path.startswith("http://") or input_path.startswith("https://")
        downloaded_temp_file = None
        original_video_path = None # Track original video for dubbing

        if is_url:
            logger.info("URL detected. Initiating download...")
            try:
                # If output is video, try to download video source
                downloaded_temp_file = download_content(input_path, Config.TEMP_DIR, prefer_video=is_video_output)
                input_file = downloaded_temp_file
                # YouTube downloads are usually video (webm/mp4) or audio (m4a/mp3). 
                # If it's a video container, we can use it for dubbing.
                if input_file.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
                    original_video_path = str(input_file)
            except Exception as e:
                raise RuntimeError(f"Could not download content from URL: {e}")
        else:
            input_file = Path(input_path)
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Check if local input is video
            if input_file.suffix.lower() in [".mp4", ".mov", ".mkv", ".avi", ".webm"]:
                original_video_path = str(input_file)

        logger.info(f"Starting processing for {input_file}")
        
        try:
            # 1. Load Audio
            # Pydub automatically handles video files by extracting the audio track (using ffmpeg)
            if original_video_path:
                logger.info(f"Video file detected: {Path(original_video_path).name}. Extracting audio track...")
                
            original_audio = self.audio_processor.load_audio(str(input_file))
            total_duration = len(original_audio) / 1000.0
            
            # 2. Strategy Decision: Split into chunks
            # The thinking model supports large context, but chunking ensures reliable timestamp alignment
            # and granular error handling. Default chunk size is 300s (5 min).
            chunks_data = self.audio_processor.detect_speech_intervals(original_audio)
            
            translated_segments_data = []
            
            for i, chunk_info in enumerate(chunks_data):
                chunk = chunk_info['audio']
                chunk_start = chunk_info['start']
                chunk_duration = len(chunk) / 1000.0
                logger.info(f"Processing chunk {i+1}/{len(chunks_data)} ({chunk_duration:.2f}s)...")
                
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
                        logger.info(f"Sending audio chunk to Gemini (Mode: {mode})...")
                        
                        translated_bytes = self.gemini_client.translate_audio(
                            str(temp_chunk_path), 
                            target_lang, 
                            duration_hint_sec=chunk_duration,
                            voice_name=voice_name,
                            mode=mode
                        )
                        
                        # Write response to temp file to load back into pydub
                        
                        # Detect format based on header
                        # Gemini often returns WAV (RIFF header) even if we requested AUDIO (default)
                        header_hex = translated_bytes[:32].hex()
                        
                        # --- Format Detection & Handling ---
                        # Case 1: Standard WAV (RIFF)
                        if translated_bytes.startswith(b'RIFF'):
                            ext = ".wav"
                        # Case 2: Raw PCM (e.g., audio/L16;codec=pcm;rate=24000)
                        # The DEBUG logs showed MimeType: audio/L16;codec=pcm;rate=24000
                        # And header all zeros (silence).  We need to wrap this in a WAV container.
                        elif len(translated_bytes) > 0 and not translated_bytes.startswith(b'ID3') and not translated_bytes.startswith(b'\xFF\xFB'):
                             # Assuming 24000Hz, 1 channel, 16-bit based on log "audio/L16;codec=pcm;rate=24000"
                             # Ideally we should pass the mimetype back from gemini.py, but for now we infer.
                             import wave
                             
                             ext = ".wav"
                             # Wrap raw PCM in WAV
                             # We'll create a new simplified bytes object with WAV header
                             # Wrap raw PCM in WAV container immediately before saving to simplify loading.
                             
                             logger.info("Raw PCM detected, wrapping in WAV container...")
                             
                             pcm_path = Config.TEMP_DIR / f"temp_raw_{i}.pcm"
                             wav_path = Config.TEMP_DIR / f"translated_chunk_{i}.wav"
                             
                             # Parameters matched to "audio/L16;codec=pcm;rate=24000"
                             with wave.open(str(wav_path), 'wb') as wav_file:
                                 wav_file.setnchannels(1)
                                 wav_file.setsampwidth(2) # 16-bit = 2 bytes
                                 wav_file.setframerate(24000)
                                 wav_file.writeframes(translated_bytes)
                                 
                             # Modify temp_out_path to point to this new wav file
                             # Skip the standard write below by setting a flag since we handled it here.
                             temp_out_path = wav_path
                             ext = ".wav" # signal for logging
                             
                             # SKIP writing translated_bytes again below, we already wrote the WAV
                             # To avoid double write, we can use a flag
                             wrote_file_already = True

                        else:
                            ext = ".mp3" # Assume MP3 otherwise, or let pydub/ffmpeg figure it out if we were ignoring extension errors.
                            wrote_file_already = False
                            
                        # Debug print to ensure code is running
                        print(f"DEBUG: Chunk {i} received {len(translated_bytes)} bytes.", flush=True)
                        print(f"DEBUG: Detected extension: {ext}", flush=True)
                        # print(f"DEBUG: Header (hex): {header_hex}", flush=True)

                        if not locals().get("wrote_file_already", False):
                            temp_out_path = Config.TEMP_DIR / f"translated_chunk_{i}{ext}"
                            with open(temp_out_path, "wb") as f:
                                f.write(translated_bytes)
                        
                        translated_segment = self.audio_processor.load_audio(str(temp_out_path))
                        
                        # Trim silence to improve sync
                        translated_segment = self.audio_processor.trim_silence(translated_segment)
                        
                        # Check Duration & Adjust if strictly needed (simple fallback)
                        actual_duration = len(translated_segment) / 1000.0
                        diff = abs(actual_duration - chunk_duration)
                        if diff > 0.2: # Drift threshold reduced to 0.2s for tighter sync
                            logger.warning(f"Drift detected in chunk {i}: target {chunk_duration}s, got {actual_duration}s. Attempting speed correction.")
                            translated_segment = self.audio_processor.speed_match(translated_segment, chunk_duration)
                        
                        translated_segments_data.append({
                            'audio': translated_segment,
                            'start': chunk_start
                        })
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
                            translated_segments_data.append({
                                'audio': AudioSegment.silent(duration=len(chunk)),
                                'start': chunk_start
                            })
                
                # Cleanup temp files
                if temp_chunk_path.exists(): os.remove(temp_chunk_path)
                # Cleanup potential temp files with different extensions
                for ext in [".mp3", ".wav"]:
                    temp_translated_path = Config.TEMP_DIR / f"translated_chunk_{i}{ext}"
                    if temp_translated_path.exists(): os.remove(temp_translated_path)

            # 3. Merge (Overlay on timeline)
            logger.info("Merging processed segments onto timeline...")
            
            # Create a silent track of the total duration
            final_voice = AudioSegment.silent(duration=int(total_duration * 1000))
            
            for seg_data in translated_segments_data:
                final_voice = final_voice.overlay(seg_data['audio'], position=seg_data['start'])
            
            # 4. Post-processing (Ducking)
            if ducking:
                logger.info("Applying audio ducking...")
                # We need the original full audio again, for the base
                # Ducking requires the original track (background) + new voice
                final_output = self.audio_processor.apply_ducking(original_audio, final_voice)
            else:
                final_output = final_voice

            # 5. Export
            # (is_video_output was determined at start of method)
            
            if is_video_output and original_video_path:
                logger.info(f"Output is video ({Path(output_path).name}) and original video available. Merging translated audio into video...")
                
                # We must first save the final audio to a temp file
                temp_final_audio = Config.TEMP_DIR / "final_translated_audio.wav"
                self.audio_processor.save_audio(final_output, str(temp_final_audio), format="wav")
                
                try:
                    self.audio_processor.merge_video_audio(
                        video_path=original_video_path,
                        audio_path=str(temp_final_audio),
                        output_path=output_path
                    )
                except Exception as e:
                    logger.error(f"Failed to merge video and audio: {e}")
                    logger.warning("Falling back to saving audio only to ensure output is preserved.")
                    # If fallback, we save as mp3 if original goal was video, or just use the name provided?
                    # The name provided likely ends in .mp4. Saving audio to .mp4 is valid (audio-only mp4).
                    # Or we can append .mp3. Let's try saving to the requested path first (audio only).
                    try:
                        self.audio_processor.save_audio(final_output, output_path)
                        logger.info(f"Saved audio-only to {output_path}")
                    except Exception as e2:
                        fallback_path = str(output_path) + ".mp3"
                        logger.error(f"Failed to save audio to original path: {e2}. Saving to {fallback_path}")
                        self.audio_processor.save_audio(final_output, fallback_path)

                finally:
                    if temp_final_audio.exists():
                        os.remove(temp_final_audio)
            
            else:
                 # Standard audio export
                 logger.info(f"Saving final output to {output_path}")
                 self.audio_processor.save_audio(final_output, output_path)

            logger.info("Done!")
            
        finally:
            if downloaded_temp_file and downloaded_temp_file.exists():
                logger.info(f"Cleaning up downloaded file: {downloaded_temp_file}")
                os.remove(downloaded_temp_file)
