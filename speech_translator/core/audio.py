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
    def split_on_silence_smart(
        audio: AudioSegment, 
        min_silence_len: int = 500, 
        silence_thresh_offset: int = -14, 
        keep_silence: int = 200,
        target_chunk_len_sec: int = 45
    ) -> List[AudioSegment]:
        """
        Splits audio on silence but tries to group chunks to reach a target length 
        to optimize for API context windows.
        """
        logger.info("Analyzing silence and splitting audio...")
        
        # Adjust silence threshold relative to the audio's dBFS
        silence_thresh = audio.dBFS + silence_thresh_offset
        
        chunks = split_on_silence(
            audio, 
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )

        if not chunks:
            logger.warning("No silence found, returning original audio as single chunk.")
            return [audio]

        combined_chunks = []
        current_chunk = chunks[0]
        
        target_len_ms = target_chunk_len_sec * 1000

        for i in range(1, len(chunks)):
            next_chunk = chunks[i]
            if len(current_chunk) + len(next_chunk) < target_len_ms:
                current_chunk += next_chunk
            else:
                combined_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        combined_chunks.append(current_chunk)
        logger.info(f"Audio split into {len(combined_chunks)} chunks.")
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
        # Lower the volume of the original track globally or specifically?
        # A simple approach for this context: Make original quiet, overlay voice.
        
        # Ideally, we keep original volume where voice is silent. 
        # But constructing that mask is complex. 
        # "Simple Ducking": Lower original volume by X dB for the entire duration of the overlay.
        
        ducked_original = original - abs(threshold_db) # Reduce volume
        return ducked_original.overlay(voice_over)

    @staticmethod
    def speed_match(segment: AudioSegment, target_duration_sec: float) -> AudioSegment:
        """
        Adjusts the speed of the segment to match target_duration_sec exactly.
        Requires ffmpeg.
        """
        current_duration_sec = len(segment) / 1000.0
        if current_duration_sec == 0:
            return segment

        speed_factor = current_duration_sec / target_duration_sec
        
        # Avoid extreme speed changes that sound robotic/broken
        if speed_factor < 0.5 or speed_factor > 2.0:
            logger.warning(f"Speed change required ({speed_factor:.2f}x) is too extreme. clamping.")
            speed_factor = max(0.5, min(speed_factor, 2.0))
            
        # pydub's speedup is simple but changes pitch? No, pydub doesn't support pitch-preserving speedup natively well without simple frame skipping.
        # Actually pydub.effects.speedup does chunking.
        # For high quality, we might prefer relying on Gemini's alignment, 
        # but as a fallback, pydub speedup is okay for small adjustments.
        try:
             # pydub speedup minimizes pitch shift artifacts for small changes
            return segment.speedup(playback_speed=speed_factor)
        except Exception as e:
            logger.warning(f"Could not apply speedup: {e}")
            return segment
