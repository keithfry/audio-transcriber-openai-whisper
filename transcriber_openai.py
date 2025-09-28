import torch
import numpy as np
import warnings
from typing import Optional
import whisper
import tempfile
import soundfile as sf


class OpenAIWhisperTranscriber:
    """
    Audio transcriber using OpenAI's original Whisper implementation.

    This transcriber provides high-quality speech-to-text conversion using the
    official OpenAI Whisper models. It's more reliable than HuggingFace alternatives
    and handles long audio automatically without manual chunking. Optimized for
    real-time transcription with robust error handling and audio preprocessing.

    Features:
    - Multiple model sizes for different accuracy/speed trade-offs
    - Automatic audio normalization and format handling
    - Optimized parameters for noise robustness
    - Support for multiple languages
    - Detailed segment information with timestamps
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize the OpenAI Whisper transcriber with specified model size.

        Loads the OpenAI Whisper model using the official whisper library.
        Different model sizes offer trade-offs between accuracy and speed:
        - tiny: Fastest, basic accuracy (~39 MB)
        - base: Good balance (~74 MB)
        - small: Better accuracy (~244 MB)
        - medium: High accuracy (~769 MB)
        - large: Best accuracy (~1550 MB)

        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')

        Raises:
            Exception: If model loading fails due to network, disk space, or compatibility issues
        """
        self.model_size = model_size
        print(f"Loading OpenAI Whisper {model_size} model...")

        try:
            self.model = whisper.load_model(model_size)
            print(f"✅ OpenAI Whisper {model_size} loaded successfully")
        except Exception as e:
            print(f"❌ Error loading OpenAI Whisper: {e}")
            raise

    def transcribe_audio(self, audio_data: np.ndarray, language: str = "en") -> dict:
        """
        Transcribe numpy audio data using OpenAI Whisper with optimized settings.

        Processes raw audio data through OpenAI's Whisper model for speech recognition.
        Handles audio normalization, applies noise filtering parameters, and returns
        structured results including text, segments, and confidence metrics. Uses
        temperature=0 for deterministic results and optimized thresholds for quality.

        Args:
            audio_data (np.ndarray): Audio data as float32 numpy array (16kHz sample rate)
            language (str): Target language code (default: 'en' for English)

        Returns:
            dict: Transcription results containing:
                - text (str): Complete transcribed text
                - segments (list): Detailed segment information with timestamps
                - language (str): Detected or specified language

        Note:
            Automatically handles audio longer than 30 seconds without chunking.
            Uses optimized parameters for real-time transcription quality.
        """
        if len(audio_data) == 0:
            return {"text": "", "segments": []}

        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val

            print(f"🎵 Audio for OpenAI Whisper: {len(audio_data)} samples, {len(audio_data)/16000:.2f}s")
            print(f"📊 Energy: {np.sqrt(np.mean(audio_data**2)):.4f}, Max: {np.max(np.abs(audio_data)):.4f}")

            # Transcribe with OpenAI Whisper
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                result = self.model.transcribe(
                    audio_data,
                    language=language,
                    task="transcribe",
                    verbose=False,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                )

            # Extract text
            transcribed_text = result.get("text", "").strip()

            return {
                "text": transcribed_text,
                "segments": result.get("segments", []),
                "language": result.get("language", language)
            }

        except Exception as e:
            print(f"❌ OpenAI Whisper transcription error: {e}")
            return {"text": f"[OpenAI Whisper Error: {str(e)}]", "segments": []}

    def transcribe_file(self, audio_file_path: str, language: str = "en") -> dict:
        """
        Transcribe audio directly from a file path using OpenAI Whisper.

        Loads and transcribes audio files without intermediate processing.
        Supports various audio formats (WAV, MP3, M4A, etc.) and automatically
        handles format conversion and sample rate adjustment. More efficient
        than loading audio into memory first for large files.

        Args:
            audio_file_path (str): Path to audio file to transcribe
            language (str): Target language code (default: 'en' for English)

        Returns:
            dict: Transcription results containing:
                - text (str): Complete transcribed text
                - segments (list): Detailed segment information with timestamps
                - language (str): Detected or specified language

        Raises:
            Exception: If file doesn't exist, format unsupported, or transcription fails
        """
        try:
            result = self.model.transcribe(
                audio_file_path,
                language=language,
                task="transcribe",
                verbose=False,
                temperature=0.0,
                condition_on_previous_text=False,
            )

            return {
                "text": result.get("text", "").strip(),
                "segments": result.get("segments", []),
                "language": result.get("language", language)
            }
        except Exception as e:
            print(f"❌ File transcription error: {e}")
            return {"text": f"[File Transcription Error: {str(e)}]", "segments": []}

    def get_model_info(self) -> dict:
        """
        Get information about the loaded Whisper model and its configuration.

        Returns metadata about the current model instance including model type,
        size, and expected sample rate. Useful for logging, debugging, and
        ensuring compatibility with audio processing pipeline.

        Returns:
            dict: Model information containing:
                - model_type (str): 'OpenAI Whisper' identifier
                - model_size (str): Size variant of loaded model
                - sample_rate (int): Expected audio sample rate (16000 Hz)
        """
        return {
            "model_type": "OpenAI Whisper",
            "model_size": self.model_size,
            "sample_rate": 16000
        }