import pytest
from unittest.mock import MagicMock
import sys
from pathlib import Path

# Add project root to path to ensure modules are found
sys.path.append(str(Path(__file__).parent.parent))

from speech_translator.config import Config

@pytest.fixture
def mock_gemini_client(mocker):
    """Mocks the Google Gemini Client."""
    # Mock the entire google.genai module to prevent actual network calls
    mock_genai = mocker.patch("speech_translator.core.gemini.genai")
    
    # Create the client mock
    mock_client_instance = MagicMock()
    mock_genai.Client.return_value = mock_client_instance
    
    # Mock models.list
    mock_client_instance.models.list.return_value = [
        MagicMock(name="models/gemini-2.0-flash-exp"),
        MagicMock(name="models/gemini-2.0-flash-thinking-exp")
    ]
    
    # Mock generate_content for Translation calls
    # We set up a default return value, but individual tests can override this
    mock_response = MagicMock()
    mock_response.text = '{"text": "Translated text", "category": "Woman"}'
    # Mock candidates for audio response
    mock_candidate = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data.data = b"fake_audio_bytes"
    mock_candidate.content.parts = [mock_part]
    mock_response.candidates = [mock_candidate]
    
    mock_client_instance.models.generate_content.return_value = mock_response
    
    return mock_client_instance

@pytest.fixture
def mock_audio_processor(mocker):
    """Mocks the AudioProcessor and underlying pydub/ffmpeg calls."""
    # Mock AudioSegment to avoid file IO and ffmpeg dependency during tests
    mock_audio_segment = mocker.patch("speech_translator.core.audio.AudioSegment")
    
    # Create a dummy audio content mock
    mock_audio_content = MagicMock()
    mock_audio_content.__len__.return_value = 5000 # 5 seconds
    mock_audio_content.duration_seconds = 5.0
    
    # Mock common methods
    mock_audio_content.overlay.return_value = mock_audio_content
    mock_audio_content.set_frame_rate.return_value = mock_audio_content
    mock_audio_content.export.return_value = None
    
    # Mock AudioSegment.from_file and silent
    mock_audio_segment.from_file.return_value = mock_audio_content
    mock_audio_segment.silent.return_value = mock_audio_content
    
    # Also mock the AudioProcessor class itself if we want to isolate logic
    # But often we want to test AudioProcessor logic while mocking AudioSegment
    return mock_audio_content

@pytest.fixture
def mock_config(mocker):
    """Allows temporarily overriding Config values."""
    mocker.patch.object(Config, 'GOOGLE_API_KEY', 'fake_key')
    mocker.patch.object(Config, 'TEMP_DIR', Path('/tmp/fake_dir'))
    return Config
