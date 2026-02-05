import typer
import logging
from rich.logging import RichHandler
from pathlib import Path
from typing import Optional
from speech_translator.orchestrator import TranslationOrchestrator
from speech_translator.config import Config
from speech_translator.core.gemini import GeminiClient

app = typer.Typer(help="Speech-to-Speech Translator using Gemini 2.5")

@app.command()
def translate(
    input_path: str = typer.Argument(..., help="Path to the input audio/video file OR a YouTube URL."),
    target_lang: str = typer.Option(..., "--lang", "-l", help="Target language (e.g., 'English', 'Spanish')."),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save the translated audio/video. Defaults to [input]_translated.[mp3|mp4]"),
    ducking: bool = typer.Option(False, "--ducking", "-d", help="Apply auto-ducking to mix translated voice with original background."),
    voice: str = typer.Option("Kore", "--voice", help="TTS Voice (Puck, Charon, Kore, Fenrir, Aoede)"),
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
        allowed_voices = ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
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
                # For URL, we can't easily guess the filename yet, so we'll let the orchestrator handle valid naming?
                # Actually, orchestrator takes a string path. 
                # CLI usually defines the output. 
                # If URL, we default to "output_translated.mp4" if we can't guess name?
                # Or better: derive from Orchestrator? No, Orchestrator expects output_path.
                # Let's set a generic default for URL if not specified.
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
            voice_name=voice
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
