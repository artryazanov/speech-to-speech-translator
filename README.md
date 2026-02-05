# Speech-to-Speech Translator

Transform your audio content into any language while preserving the original voice, emotion, and intonation. Powered by Google Gemini 2.5 Flash Native Audio.

## 🚀 Features

- **Native Audio Processing**: Uses Gemini 2.5 Flash for direct audio-to-audio translation without intermediate text steps.
- **Voice & Emotion Preservation**: Keeps the original speaker's timbre and emotional delivery.
- **Timestamp Alignment**: Automatically syncs the translated speech duration to match the original video/audio.
- **Smart Chunking**: Handles long files by intelligently splitting based on silence.
- **Video Input Support**: Automatically extracts audio from video files (mp4, mov, mkv, etc.) for translation.
- **YouTube Support**: Download and translate audio directly from YouTube URL.
- **Audio Ducking**: (Optional) Mixes the translated voice with the original background audio.

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/artryazanov/speech-to-speech-translator.git
   cd speech-to-speech-translator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # OR
   pip install .
   # OR if using PDM
   pdm install
   ```

3. **FFmpeg is required** for audio processing.
   - Ubuntu: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from ffmpeg.org and add to PATH.

4. **Set up your API Key:**

   1. Go to [Google AI Studio](https://aistudio.google.com/).
   2. Sign in with your Google account.
   3. Click on **"Get API key"** (top left).
   4. Click **"Create API key"** (or use an existing project).
   5. Copy `.env.example` to `.env` and paste your key:
      ```bash
      cp .env.example .env
      # Open .env and set GOOGLE_API_KEY=your_copied_key_here
      ```

## 🎙️ Usage

### Zero-Config CLI

Translate a file to English:
```bash
python -m speech_translator.cli input.mp3 --lang "English" --output output.mp3
```

Translate a YouTube video:
```bash
python -m speech_translator.cli "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --lang "French"
```

### Advanced Options

```bash
python -m speech_translator.cli input.mp3 \
  --lang "Spanish" \
  --output result.mp3 \
  --ducking \
```

### Docker Usage

You can run the tool without installing dependencies on your host machine using Docker.

1. **Build the image**:
   ```bash
   docker compose build
   ```

2. **Run a translation**:
   Place your input file (e.g., `input.mp3`) in the project directory.
   ```bash
   docker compose run --rm translator translate input.mp3 --lang "English" --output output.mp3
   ```
   *Note: The current directory is mounted to `/app/data` inside the container, so input and output files are read/written directly to your host folder.*

3. **Translate a YouTube video**:
   ```bash
   docker compose run --rm translator "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --lang "French" --output youtube_audio.mp3
   ```

## 🏗️ Architecture

This project uses a modular "Service" pattern:

- **Core Audio**: `pydub` handles splitting, merging, and effects.
- **AI Engine**: `google-generativeai` interfaces with Gemini 2.5.
- **Orchestrator**: Manages the pipeline of Chunk -> Translate -> Merge.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
