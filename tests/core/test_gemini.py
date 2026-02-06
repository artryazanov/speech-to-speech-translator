import pytest
from unittest.mock import MagicMock, patch
from speech_translator.core.gemini import GeminiClient
from google.genai import types

class TestGeminiClient:
    
    def test_init(self, mock_config, mock_gemini_client):
        """Test initialization of GeminiClient."""
        client = GeminiClient()
        assert client.client is not None
        assert client.thinking_model == mock_config.THINKING_MODEL
        assert client.tts_model == mock_config.TTS_MODEL

    def test_list_models(self, mock_gemini_client):
        """Test listing models."""
        client = GeminiClient()
        models = client.list_models()
        assert len(models) == 2
        mock_gemini_client.models.list.assert_called_once()

    def test_translate_audio_auto_voice(self, mock_gemini_client, tmp_path):
        """Test Step 1: Translation with Auto voice selection."""
        client = GeminiClient()
        
        # Create a dummy audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"dummy_audio_content")
        
        # Mock response for Step 1 (Thinking Model)
        mock_response_step1 = MagicMock()
        mock_response_step1.text = '{"category": "Young Man", "text": "Hello world"}'
        
        # Mock response for Step 2 (TTS Model)
        mock_response_step2 = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"translated_audio_bytes"
        mock_response_step2.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        # Configure side_effect to return different responses for the two calls
        mock_gemini_client.models.generate_content.side_effect = [
            mock_response_step1, # Step 1: Analyze & Translate
            mock_response_step2  # Step 2: TTS
        ]
        
        result = client.translate_audio(str(audio_file), "English", voice_name="Auto", duration_hint_sec=5.0)
        
        # Verify result is the audio bytes from Step 2
        assert result == b"translated_audio_bytes"
        
        # Verify API calls
        assert mock_gemini_client.models.generate_content.call_count == 2
        
        # Check Step 1 call args (Speaker Analysis)
        args_step1, kwargs_step1 = mock_gemini_client.models.generate_content.call_args_list[0]
        assert kwargs_step1['config'].response_mime_type == "application/json"
        
        # Check Step 2 call args (TTS)
        args_step2, kwargs_step2 = mock_gemini_client.models.generate_content.call_args_list[1]
        speech_config = kwargs_step2['config'].speech_config
        # "Young Man" maps to "Puck" in our logic
        assert speech_config.voice_config.prebuilt_voice_config.voice_name == "Puck"

    def test_translate_audio_manual_voice(self, mock_gemini_client, tmp_path):
        """Test Step 1: Translation with specific voice."""
        client = GeminiClient()
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"dummy")
        
        # Mock response Step 1
        mock_response_1 = MagicMock()
        mock_response_1.text = "Just translation"
        
        # Mock response Step 2
        mock_response_2 = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"audio"
        mock_response_2.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        
        mock_gemini_client.models.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        client.translate_audio(str(audio_file), "Russian", voice_name="Fenrir", duration_hint_sec=5.0)
        
        # Verify TTS used "Fenrir"
        args_2, kwargs_2 = mock_gemini_client.models.generate_content.call_args_list[1]
        voice_name = kwargs_2['config'].speech_config.voice_config.prebuilt_voice_config.voice_name
        assert voice_name == "Fenrir"

    def test_translate_audio_json_error(self, mock_gemini_client, tmp_path):
        """Test fallback when JSON parsing fails in Auto mode."""
        client = GeminiClient()
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"dummy")
        
        # Mock invalid JSON response
        mock_response_1 = MagicMock()
        mock_response_1.text = "Not JSON data" # Should trigger value error or fallback? logic says fallback
        
        mock_response_2 = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"audio"
        mock_response_2.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        
        mock_gemini_client.models.generate_content.side_effect = [mock_response_1, mock_response_2]
        
        client.translate_audio(str(audio_file), "En", voice_name="Auto", duration_hint_sec=5.0)
        
        # Should catch JSONDecodeError and fall back to "Kore"
        args_2, kwargs_2 = mock_gemini_client.models.generate_content.call_args_list[1]
        voice_name = kwargs_2['config'].speech_config.voice_config.prebuilt_voice_config.voice_name
        assert voice_name == "Kore"

    def test_translate_api_failure(self, mock_gemini_client, tmp_path):
        """Test handling of API errors."""
        client = GeminiClient()
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"dummy")
        
        mock_gemini_client.models.generate_content.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            client.translate_audio(str(audio_file), "En", duration_hint_sec=5.0)
