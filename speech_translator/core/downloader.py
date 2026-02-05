import logging
import yt_dlp
from pathlib import Path

logger = logging.getLogger(__name__)

def download_content(url: str, output_dir: Path) -> Path:
    """
    Downloads audio from a given URL using yt-dlp.
    Returns the path to the downloaded file.
    """
    logger.info(f"Downloading content from: {url}")
    
    # yt-dlp options: download best audio, save with title and ID
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(title)s [%(id)s].%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'overwrites': True,
        # Explicitly tell yt-dlp where to find node to avoid warnings/errors
        # The Dockerfile creates a link at /usr/bin/node
        # Format must be {'runtime': {'key': 'value'}}
        'js_runtimes': {'node': {'path': '/usr/bin/node'}},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info to get the filename
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            
            downloaded_path = Path(filename)
            logger.info(f"Downloaded file to: {downloaded_path}")
            return downloaded_path
            
    except Exception as e:
        logger.error(f"Failed to download URL: {e}")
        raise
