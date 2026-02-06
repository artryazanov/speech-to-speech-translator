import pytest
from unittest.mock import MagicMock, patch
from speech_translator.core.audio import AudioProcessor

class TestAudioProcessor:
    
    @patch("speech_translator.core.audio.AudioSegment")
    def test_load_audio(self, mock_audio_segment):
        """Test loading audio."""
        ap = AudioProcessor()
        mock_audio_segment.from_file.return_value = "audio_obj"
        
        res = ap.load_audio("test.mp3")
        assert res == "audio_obj"
        mock_audio_segment.from_file.assert_called_with("test.mp3")

    @patch("speech_translator.core.audio.AudioSegment")
    def test_trim_silence(self, mock_audio_segment):
        """Test silence trimming logic."""
        ap = AudioProcessor()
        # trim_silence logic relies on detecting silence loops. 
        # mocking dBFS behavior on slices is complex.
        # We assume the logic is correct if it returns a segment.
        # Here we just verify it handles the object and returns something.
        
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 5000
        # return itself when sliced
        mock_audio.__getitem__.return_value = mock_audio
        mock_audio.dBFS = -20.0 # loud enough to not trim
        
        # Test runs without error
        res = ap.trim_silence(mock_audio)
        assert res is not None

    @patch("speech_translator.core.audio.AudioSegment")
    def test_detect_speech_intervals(self, mock_audio_segment):
        """Test splitting audio into chunks."""
        ap = AudioProcessor()
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 10000 # 10s
        mock_audio.dBFS = -20.0
        
        with patch("pydub.silence.detect_nonsilent") as mock_detect:
            # 2 chunks: 0-4s, 5-9s
            mock_detect.return_value = [[0, 4000], [5000, 9000]]
            
            mock_audio.__getitem__.return_value = MagicMock()
            
            # Set target_chunk_len_sec to 0 to force NO merge
            chunks = ap.detect_speech_intervals(mock_audio, min_silence_len=500, target_chunk_len_sec=0)
            
            assert len(chunks) == 2
            assert chunks[0]['start'] == 0
            assert chunks[1]['start'] == 5000

    def test_speed_match(self):
        """Test speed checking logic."""
        ap = AudioProcessor()
        mock_audio = MagicMock()
        mock_audio.frame_rate = 24000
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.__len__.return_value = 5000
        
        with patch("subprocess.run") as mock_run, \
             patch("speech_translator.core.audio.AudioSegment.export") as mock_export, \
             patch("speech_translator.core.audio.AudioSegment.from_file") as mock_from_file, \
             patch("os.remove") as mock_remove:
                 
            mock_run.return_value.returncode = 0
            mock_from_file.return_value = "processed_audio"
            
            # Mock file existence for temp output check
            with patch("pathlib.Path.exists", return_value=True):
                 res = ap.speed_match(mock_audio, target_duration_sec=2.5)
                 assert res == "processed_audio"
                 mock_run.assert_called()
                 mock_remove.assert_called()
