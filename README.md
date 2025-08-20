# Audio Transcription Web App

## Overview
- Flask-based web application for transcribing audio files to text.
- Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for speech-to-text and [spaCy](https://spacy.io/) for sentence segmentation.
- Supports multiple audio formats: `.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`.
- Provides a web interface for uploading audio files and viewing transcriptions.

## Features
- Upload one or more audio files for transcription.
- Automatic language detection and probability score.
- Transcription output is split into sentences for readability.
- Health check endpoint for monitoring model status.
- Handles large files (up to 100MB).
- Logs processing steps and errors for easier debugging.

## Requirements
- Python 3.8+
- CUDA-enabled GPU (optional, for faster inference)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [spaCy](https://spacy.io/)
- Flask
- Other dependencies in `requirements.txt`

## Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Run the app**
   ```sh
   python app.py
   ```

4. **Access the web interface**
   - Open [http://localhost:5000](http://localhost:5000) in your browser.

## API Endpoints

- `/`: Web interface for uploading audio files.
- `/transcribe` : POST endpoint for audio file(s) upload and transcription (returns JSON).
- `/health`: Health check endpoint (returns model status).

## Notes
- If CUDA is not available, the app will automatically fall back to CPU.
- Temporary files are cleaned up after processing.
- Error handling and logging are implemented for robustness.

## License
[MIT](LICENSE) (or your chosen license)
