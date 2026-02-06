import pytest
from unittest.mock import MagicMock, call, patch, ANY
from speech_translator.orchestrator import TranslationOrchestrator
from speech_translator.config import Config
from pathlib import Path

class TestOrchestrator:

    @pytest.fixture
    def orchestrator(self, mock_gemini_client, mock_audio_processor):
        # Patch AudioSegment in orchestrator to return our mock
        with patch("speech_translator.orchestrator.AudioSegment") as mock_as, \
             patch("speech_translator.orchestrator.AudioProcessor") as mock_ap_class:
            
            # Setup AudioSegment.silent to return our mock audio
            mock_as.silent.return_value = mock_audio_processor
            
            mock_ap_instance = mock_ap_class.return_value
            # Setup common mock returns
            mock_ap_instance.load_audio.return_value = mock_audio_processor
            mock_ap_instance.detect_speech_intervals.return_value = [
                {'audio': mock_audio_processor, 'start': 0, 'end': 5000}
            ]
            mock_ap_instance.trim_silence.return_value = mock_audio_processor
            mock_ap_instance.speed_match.return_value = mock_audio_processor
            mock_ap_instance.apply_ducking.return_value = mock_audio_processor
            
            orch = TranslationOrchestrator()
            orch.gemini_client = mock_gemini_client
            orch.audio_processor = mock_ap_instance
            yield orch

    def test_process_flow_video(self, orchestrator, mock_gemini_client, tmp_path):
        """Test full flow for video input."""
        # Setup input
        input_file = tmp_path / "input.mp4"
        input_file.touch() # Create dummy file
        output_file = tmp_path / "output.mp4"
        
        # Mock translate response
        mock_gemini_client.translate_audio.return_value = b"RIFF_fake_wav_header"
        
        orchestrator.process(str(input_file), str(output_file), "Spanish", ducking=True)
        
        # Verify Audio Loaded
        orchestrator.audio_processor.load_audio.assert_called()
        
        # Verify Translation Called
        mock_gemini_client.translate_audio.assert_called()
        
        # Verify Ducking Applied
        orchestrator.audio_processor.apply_ducking.assert_called()
        
        # Verify Video Merge
        orchestrator.audio_processor.merge_video_audio.assert_called_with(
            video_path=str(input_file),
            audio_path=ANY,
            output_path=str(output_file)
        )

    def test_process_flow_audio_only(self, orchestrator, mock_gemini_client, tmp_path):
        """Test flow for audio-only export."""
        input_file = tmp_path / "input.mp3"
        input_file.touch()
        output_file = tmp_path / "output.mp3"
        
        mock_gemini_client.translate_audio.return_value = b"fake_mp3_data"
        
        orchestrator.process(str(input_file), str(output_file), "French", ducking=False)
        
        # Verify Save Audio called instead of merge video
        orchestrator.audio_processor.save_audio.assert_called_with(ANY, str(output_file))
        orchestrator.audio_processor.merge_video_audio.assert_not_called()

    def test_process_retry_logic(self, orchestrator, mock_gemini_client, tmp_path):
        """Test that it retries on API failure."""
        input_file = tmp_path / "input.mp3"
        input_file.touch()
        
        # First call fails, second succeeds
        mock_gemini_client.translate_audio.side_effect = [
            Exception("429 Rate Limit"),
            b"success_audio"
        ]
        
        with patch("time.sleep") as mock_sleep: # Don't actually sleep
            orchestrator.process(str(input_file), str(tmp_path/"out.mp3"), "En")
            
            assert mock_gemini_client.translate_audio.call_count == 2
            mock_sleep.assert_called()

    def test_process_url_download(self, orchestrator, tmp_path):
        """Test handling of URL inputs."""
        url = "http://example.com/video.mp4"
        output = str(tmp_path / "out.mp4")
        
        # Mock downloader
        with patch("speech_translator.orchestrator.download_content") as mock_dl:
            mock_dl.return_value = tmp_path / "downloaded.mp4"
            (tmp_path / "downloaded.mp4").touch()
            
            orchestrator.process(url, output, "En")
            
            mock_dl.assert_called_with(url, ANY, prefer_video=True)
