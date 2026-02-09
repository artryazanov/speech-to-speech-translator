import typer
import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
from enum import Enum
from pathlib import Path
from typing import Optional
from speech_translator.orchestrator import TranslationOrchestrator
from speech_translator.config import Config
from speech_translator.core.gemini import GeminiClient

# Create Enum for modes
class TranslationMode(str, Enum):
    MONOLOGUE = "monologue"
    DIALOGUE = "dialogue"

app = typer.Typer(help="Speech-to-Speech Translator using Gemini models")

@app.command()
def translate(
    input_path: str = typer.Argument(..., help="Path to the input audio/video file OR a YouTube URL."),
    target_lang: str = typer.Option(..., "--lang", "-l", help="Target language (e.g., 'English', 'Spanish')."),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save the translated audio/video. Defaults to [input]_translated.[mp3|mp4]"),
    ducking: bool = typer.Option(False, "--ducking", "-d", help="Apply auto-ducking to mix translated voice with original background."),
    voice: str = typer.Option("Auto", "--voice", help="TTS Voice (Auto, Puck, Charon, Kore, Fenrir, Aoede). Ignored in Dialogue mode."),
    mode: TranslationMode = typer.Option(TranslationMode.MONOLOGUE, "--mode", "-m", help="Processing mode: monologue (single speaker) or dialogue (multi-speaker detection)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
):
    """
    Translates an audio or video file to another language preserving voice and intonation.
    If a video file is provided, the audio track will be extracted and translated.
    If the output path is not specified, it defaults to the input filename with _translated suffix.
    """
    # Setup Logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)]
    )
    
    try:
        # Validate Config
        Config.validate()
        
        # Validate Voice
        allowed_voices = ["Auto", "Puck", "Charon", "Kore", "Fenrir", "Aoede"]
        if voice not in allowed_voices:
            typer.secho(f"Warning: '{voice}' is not a standard voice. Allowed: {allowed_voices}. Proceeding anyway...", fg=typer.colors.YELLOW)

        # Infer Output Path if not provided
        if output_path is None:
            # Check if input looks like a video (URL or extension)
            is_video_input = False
            input_lower = input_path.lower()
            video_extensions = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
            
            if input_lower.startswith("http://") or input_lower.startswith("https://"):
                # Assume URLs (YouTube) are videos by default unless proven otherwise
                is_video_input = True
                # Default to "downloaded_video" for URLs since filename inference
                # happens later or is not guaranteed.
                base_name = "downloaded_video"
            else:
                input_p = Path(input_path)
                if input_p.suffix.lower() in video_extensions:
                    is_video_input = True
                base_name = input_p.stem

            ext = ".mp4" if is_video_input else ".mp3"
            output_path = Path(f"{base_name}_translated{ext}")
            typer.secho(f"No output path provided. Defaulting to: {output_path}", fg=typer.colors.CYAN)

        orchestrator = TranslationOrchestrator()
        orchestrator.process(
            input_path=str(input_path),
            output_path=str(output_path),
            target_lang=target_lang,
            ducking=ducking,
            voice_name=voice,
            mode=mode.value
        )
        
        typer.secho(f"Success! Translation saved to {output_path}", fg=typer.colors.GREEN, bold=True)
        
    except ValueError as val_err:
        typer.secho(f"Configuration Error: {val_err}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"An error occurred: {e}", fg=typer.colors.RED)
        if verbose:
            logger.exception("Traceback:")
        raise typer.Exit(code=1)

@app.command(name="list-models")
def list_models():
    """
    Lists all available Gemini models.
    """
    try:
        Config.validate()
        client = GeminiClient()
        models = client.list_models()
        
        typer.secho("Available Gemini Models:", fg=typer.colors.BLUE, bold=True)
        for model in models:
            typer.echo(f"- {model.name} ({model.display_name})")
            
    except Exception as e:
        typer.secho(f"Error listing models: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
