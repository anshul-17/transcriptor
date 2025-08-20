from flask import Flask, request, jsonify, render_template
import os
import spacy
from faster_whisper import WhisperModel
import tempfile
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize models
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("SpaCy model loaded successfully")
except Exception as e:
    logger.error(f"Error loading SpaCy model: {e}")
    nlp = None

try:
    model = WhisperModel("small", device="cuda", compute_type="float16")
    logger.info("Whisper model loaded successfully with CUDA")
except Exception as e:
    logger.warning(f"CUDA not available, falling back to CPU: {e}")
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully with CPU")
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        model = None

# Supported audio formats
ALLOWED_EXTENSIONS = {'.mp3', '.m4a', '.wav', '.flac', '.ogg'}


def allowed_file(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if not model or not nlp:
        return jsonify({'error': 'Models not loaded properly'}), 500

    results = []

    try:
        # Process each uploaded file
        for key in request.files:
            file = request.files[key]
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                logger.info(f"Processing file: {filename}")

                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                    file.save(temp_file.name)
                    temp_path = temp_file.name

                try:
                    # Transcribe the audio
                    segments, info = model.transcribe(temp_path)
                    transcription_text = " ".join([seg.text for seg in segments])

                    logger.info(f"Transcription completed for {filename}")

                    # Break into sentences using spaCy
                    if transcription_text.strip():
                        doc = nlp(transcription_text)
                        sentences = []
                        for sent in doc.sents:
                            sentence = sent.text.strip()
                            if sentence:  # Skip empty sentences
                                sentences.append(sentence)
                    else:
                        sentences = ["No speech detected in the audio file."]

                    results.append({
                        'filename': filename,
                        'sentences': sentences,
                        'language': info.language,
                        'language_probability': round(info.language_probability, 2)
                    })

                except Exception as e:
                    logger.error(f"Error transcribing {filename}: {e}")
                    results.append({
                        'filename': filename,
                        'sentences': [f"Error transcribing file: {str(e)}"],
                        'language': 'unknown',
                        'language_probability': 0
                    })

                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Could not delete temp file: {e}")

        return jsonify(results)

    except Exception as e:
        logger.error(f"Unexpected error during transcription: {e}")
        return jsonify({'error': 'An unexpected error occurred during transcription'}), 500


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'whisper_loaded': model is not None,
        'spacy_loaded': nlp is not None
    })


if __name__ == '__main__':
    print("Starting Audio Transcription Server...")
    print("Available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    app.run(host='0.0.0.0', port=5000, debug=True)