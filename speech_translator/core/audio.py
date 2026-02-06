import logging
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

class AudioProcessor:
    @staticmethod
    def load_audio(path: str) -> AudioSegment:
        """Loads an audio file from path."""
        try:
            return AudioSegment.from_file(path)
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    @staticmethod
    def save_audio(segment: AudioSegment, path: str, format: str = "mp3"):
        """Saves an audio segment to path."""
        segment.export(path, format=format)

    @staticmethod
    def detect_speech_intervals(
        audio: AudioSegment, 
        min_silence_len: int = 500, 
        silence_thresh_offset: int = -14, 
        target_chunk_len_sec: int = 300
    ) -> List[dict]:
        """
        Detects speech intervals and groups them into chunks of approximately chunk_len_sec.
        Returns a list of dictionaries: {'audio': AudioSegment, 'start': int (ms), 'end': int (ms)}
        """
        from pydub.silence import detect_nonsilent
        
        logger.info("Detecting speech intervals...")
        
        # Adjust silence threshold relative to the audio's dBFS
        silence_thresh = audio.dBFS + silence_thresh_offset
        
        # detect_nonsilent returns list of [start, end] pairs in ms
        nonsilent_ranges = detect_nonsilent(
            audio, 
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        if not nonsilent_ranges:
            logger.warning("No speech detected, returning original audio as single chunk.")
            return [{'audio': audio, 'start': 0, 'end': len(audio)}]

        combined_chunks = []
        
        target_len_ms = target_chunk_len_sec * 1000
        
        # Start with the first range
        current_start, current_end = nonsilent_ranges[0]
        
        for i in range(1, len(nonsilent_ranges)):
            next_start, next_end = nonsilent_ranges[i]
            
            # Calculate potential new length if we merge up to next_end
            # We include the silence between current_end and next_start
            potential_len = next_end - current_start
            
            if potential_len < target_len_ms:
                # Merge: extend current endpoint
                current_end = next_end
            else:
                # Commit current chunk
                chunk_audio = audio[current_start:current_end]
                combined_chunks.append({
                    'audio': chunk_audio, 
                    'start': current_start,
                    'end': current_end
                })
                
                # Start new chunk
                current_start = next_start
                current_end = next_end
        
        # Append the final chunk
        chunk_audio = audio[current_start:current_end]
        combined_chunks.append({
            'audio': chunk_audio, 
            'start': current_start,
            'end': current_end
        })
        
        logger.info(f"Audio split into {len(combined_chunks)} intervals/chunks.")
        return combined_chunks

    @staticmethod
    def merge_segments(segments: List[AudioSegment], crossfade: int = 0) -> AudioSegment:
        """Merges a list of audio segments into one."""
        if not segments:
            return AudioSegment.empty()
        
        final_audio = segments[0]
        for segment in segments[1:]:
            final_audio = final_audio.append(segment, crossfade=crossfade)
            
        return final_audio

    @staticmethod
    def apply_ducking(original: AudioSegment, voice_over: AudioSegment, threshold_db: int = -15) -> AudioSegment:
        """
        Overlays the voice_over onto the original audio, ducking (lowering volume) 
        of the original where the voice_over exists.
        
        Note: This implementation is simple. For 'true' ducking we would track 
        volume of voice_over over time. Here we assume constant ducking for the duration.
        """
        # Strategy: "Simple Ducking"
        # Reduce the volume of the original track by the threshold for the entire duration.
        # This avoids complex masking while ensuring the voice-over is clearly audible.
        
        ducked_original = original - abs(threshold_db) # Reduce volume
        return ducked_original.overlay(voice_over)

    @staticmethod
    def speed_match(segment: AudioSegment, target_duration_sec: float) -> AudioSegment:
        """
        Adjusts the speed of the segment to match target_duration_sec exactly using ffmpeg's atempo filter.
        This allows both high-quality slow-down and speed-up without pitch shifting.
        """
        current_duration_sec = len(segment) / 1000.0
        if current_duration_sec == 0 or target_duration_sec <= 0:
            return segment

        speed_factor = current_duration_sec / target_duration_sec
        logger.info(f"Applying speed correction: {current_duration_sec:.2f}s -> {target_duration_sec:.2f}s (Factor: {speed_factor:.2f}x)")

        # Sanity check limits
        if speed_factor < 0.25 or speed_factor > 4.0:
            logger.warning(f"Speed change ({speed_factor:.2f}x) too extreme. Clamping to 0.5x - 2.0x for safety.")
            speed_factor = max(0.5, min(speed_factor, 2.0))
            
        if abs(speed_factor - 1.0) < 0.01:
            return segment

        import subprocess
        import os
        from speech_translator.config import Config

        # Create temp files
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        temp_input = Config.TEMP_DIR / f"speed_input_{unique_id}.wav"
        temp_output = Config.TEMP_DIR / f"speed_output_{unique_id}.wav"
        
        try:
            segment.export(str(temp_input), format="wav")
            
            # Construct ffmpeg command
            # atempo filter supports 0.5 to 2.0. Chain them for larger changes.
            filter_str = ""
            remaining_factor = speed_factor
            
            while remaining_factor > 2.0:
                filter_str += "atempo=2.0,"
                remaining_factor /= 2.0
            while remaining_factor < 0.5:
                filter_str += "atempo=0.5,"
                remaining_factor /= 0.5
            
            filter_str += f"atempo={remaining_factor}"
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_input),
                "-filter:a", filter_str,
                "-vn", # No video
                str(temp_output)
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg speedup failed: {result.stderr}")
                return segment
                
            # Load result
            if temp_output.exists():
                processed_segment = AudioSegment.from_file(str(temp_output), format="wav")
                return processed_segment
            else:
                logger.error("FFmpeg output file not found.")
                return segment
                
        except Exception as e:
            logger.error(f"Error in speed_match: {e}")
            return segment
        finally:
            if temp_input.exists(): os.remove(temp_input)
            if temp_output.exists(): os.remove(temp_output)

    @staticmethod
    def merge_video_audio(video_path: str, audio_path: str, output_path: str):
        """
        Merges the video track from video_path with the audio track from audio_path.
        The output is saved to output_path.
        Uses ffmpeg stream mapping to avoid re-encoding video.
        """
        import subprocess
        
        logger.info(f"Merging video '{video_path}' with audio '{audio_path}' -> '{output_path}'")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",  # Copy video stream without re-encoding
            "-c:a", "aac",   # Encode audio to aac (widely supported)
            "-map", "0:v:0", # Use 1st video stream from 1st input
            "-map", "1:a:0", # Use 1st audio stream from 2nd input
            "-shortest",     # Finish when the shorter stream ends (usually audio matches video, but safety first)
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg merge failed: {result.stderr}")
                raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")
            logger.info("Video merge successful.")
        except Exception as e:
            logger.error(f"Failed to merge video and audio: {e}")
            raise
