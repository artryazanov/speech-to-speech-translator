import pytest
from unittest.mock import MagicMock, patch
import json
import io
from pydub import AudioSegment
from speech_translator.core.gemini import GeminiClient
from speech_translator.config import Config

@pytest.fixture
def mock_gemini_client(mocker):
    # Mock Config
    mocker.patch.object(Config, 'GOOGLE_API_KEY', 'fake_key')
    mocker.patch.object(Config, 'GEMINI_API_VERSION', 'v1beta')
    mocker.patch.object(Config, 'THINKING_MODEL', 'gemini-2.0-flash-thinking-exp-01-21')
    mocker.patch.object(Config, 'TTS_MODEL', 'gemini-2.0-flash-exp')
    
    # Mock genai.Client
    with patch("speech_translator.core.gemini.genai.Client") as MockClient:
        client = GeminiClient()
        client.client = MockClient.return_value
        yield client

def test_translate_audio_dialogue_mode(mock_gemini_client):
    # Mock input audio file
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_audio_bytes"
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock Response 1 (Thinking Model - Diarization)
        mock_response_1 = MagicMock()
        mock_response_1.text = json.dumps({
            "segments": [
                {"speaker": "A", "category": "Man", "text": "Hello.", "approx_duration_ratio": 0.5},
                {"speaker": "B", "category": "Woman", "text": "Hi there.", "approx_duration_ratio": 0.5}
            ]
        })
        
        # Mock Response 2 (TTS) - We expect 2 calls
        mock_response_tts = MagicMock()
        mock_part = MagicMock()
        # Mocking Raw PCM data (random bytes)
        # 1 second of 24k 16bit mono = 48000 bytes
        mock_part.inline_data.data = b"\x00\x00" * 24000 
        mock_response_tts.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        
        mock_gemini_client.client.models.generate_content.side_effect = [
            mock_response_1, # Diarization
            mock_response_tts, # TTS 1
            mock_response_tts  # TTS 2
        ]

        # Use real AudioSegment for this test to match usage of _load_audio_bytes
        # But we need to avoid pydub trying to actually use ffmpeg if not installed in mocked env?
        # The new code uses AudioSegment(data=...) which doesn't need ffmpeg for Raw PCM.
        # But export() does need ffmpeg or pydub internal encoder.
        # Let's mock AudioSegment again but allow the constructor logic or just mock _load_audio_bytes?
        # Actually better to integration test the logic in _load_audio_bytes.
        
        # Mock AudioSegment export to avoid ffmpeg call
        with patch.object(AudioSegment, 'export') as mock_export:
             mock_export.return_value = None # Just to avoid error
             
             # Mock _load_audio_bytes to return a mock segment
             with patch.object(mock_gemini_client, '_load_audio_bytes') as mock_load:
                 mock_segment = MagicMock()
                 # Make sure += works (returns self)
                 mock_segment.__add__.return_value = mock_segment
                 mock_load.return_value = mock_segment
                 
                 mock_gemini_client.translate_audio(
                    "dummy_path.mp3", 
                    "English", 
                    duration_hint_sec=10.0, 
                    mode="dialogue"
                )
                 assert mock_load.call_count == 2

def test_get_voice_for_category(mock_gemini_client):
    assert mock_gemini_client._get_voice_for_category("Young Man") == "Puck"
    assert mock_gemini_client._get_voice_for_category("Elderly Man") == "Charon"
    assert mock_gemini_client._get_voice_for_category("Man") == "Fenrir"
    assert mock_gemini_client._get_voice_for_category("Young Woman") == "Aoede"
    assert mock_gemini_client._get_voice_for_category("Woman") == "Kore"
    assert mock_gemini_client._get_voice_for_category("Unknown") == "Kore"

def test_process_dialogue_no_segments_fallback(mock_gemini_client):
    # Scenario: Model returns valid JSON but empty segments list
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = b"fake_audio_bytes"
        mock_open.return_value.__enter__.return_value = mock_file
        
        mock_response_1 = MagicMock()
        mock_response_1.text = json.dumps({"segments": []})
        
        mock_response_tts = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data.data = b"fallback_audio"
        mock_response_tts.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        
        mock_gemini_client.client.models.generate_content.side_effect = [
            mock_response_1, 
            mock_response_tts
        ]

        result = mock_gemini_client.translate_audio("path", "En", mode="dialogue", duration_hint_sec=10.0)
        
        # Should call generate_content twice (1 analysis, 1 fallback TTS)
        assert mock_gemini_client.client.models.generate_content.call_count == 2
        assert result == b"fallback_audio"
