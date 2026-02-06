import pytest
import os
from unittest.mock import patch
from speech_translator.config import Config

class TestConfig:
    
    def test_validate_missing_key(self):
        """Test validation fails when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Reset Config attributes to ensure we test fresh load/validate state if possible
            # But Config is a class with static attributes populated at import time or initialization.
            # Let's rely on Config.validate() checking env or attributes.
            
            # We must mock the attribute on the class directly if it's already loaded
            with patch.object(Config, 'GOOGLE_API_KEY', None):
                 with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                    Config.validate()

    def test_validate_success(self):
        """Test validation succeeds with key."""
        with patch.object(Config, 'GOOGLE_API_KEY', "fake_key"):
            try:
                Config.validate()
            except ValueError:
                pytest.fail("Config.validate() raised ValueError unexpectedly")
