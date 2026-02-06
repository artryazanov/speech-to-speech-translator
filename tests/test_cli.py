import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from speech_translator.cli import app

runner = CliRunner()

class TestCLI:
    
    @patch("speech_translator.cli.TranslationOrchestrator")
    def test_translate_command(self, mock_orchestrator):
        """Test the main translation command."""
        mock_instance = mock_orchestrator.return_value
        
        result = runner.invoke(app, ["translate", "input.mp3", "--output", "output.mp3", "--lang", "German"])
        
        assert result.exit_code == 0
        mock_instance.process.assert_called_once()
        args, kwargs = mock_instance.process.call_args
        assert kwargs['input_path'] == "input.mp3"
        assert kwargs['output_path'] == "output.mp3"
        assert kwargs['target_lang'] == "German"

    @patch("speech_translator.cli.GeminiClient")
    def test_list_models_command(self, mock_client):
        """Test the list-models command."""
        mock_instance = mock_client.return_value
        mock_model = MagicMock()
        mock_model.name = "models/gemini-pro"
        mock_instance.list_models.return_value = [mock_model]
        
        result = runner.invoke(app, ["list-models"])
        
        assert result.exit_code == 0
        assert "models/gemini-pro" in result.stdout
