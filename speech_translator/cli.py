import typer
import logging
from rich.logging import RichHandler
from pathlib import Path
from typing import Optional
from speech_translator.orchestrator import TranslationOrchestrator
from speech_translator.config import Config

app = typer.Typer(help="Speech-to-Speech Translator using Gemini 2.5")

@app.command()
def translate(
    input_path: Path = typer.Argument(..., exists=True, help="Path to the input audio (mp3, wav) or video (mp4, mov, mkv) file."),
    target_lang: str = typer.Option(..., "--lang", "-l", help="Target language (e.g., 'English', 'Spanish')."),
    output_path: Path = typer.Option("output.mp3", "--output", "-o", help="Path to save the translated audio."),
    ducking: bool = typer.Option(False, "--ducking", "-d", help="Apply auto-ducking to mix translated voice with original background."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
):
    """
    Translates an audio or video file to another language preserving voice and intonation.
    If a video file is provided, the audio track will be extracted and translated.
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
        
        orchestrator = TranslationOrchestrator()
        orchestrator.process(
            input_path=str(input_path),
            output_path=str(output_path),
            target_lang=target_lang,
            ducking=ducking
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

if __name__ == "__main__":
    app()
