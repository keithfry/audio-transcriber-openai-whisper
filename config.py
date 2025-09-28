import os
from typing import Optional


class Config:
    """
    Configuration management for the audio transcription application.

    Loads and validates configuration parameters from environment variables
    with sensible defaults. Handles audio settings, model selection, voice
    activity detection parameters, and application behavior options.
    Provides validation to ensure configuration values are within acceptable ranges.
    """
    def __init__(self):
        """
        Initialize configuration by loading values from environment variables.

        Reads configuration from environment variables with fallback defaults.
        Supports both OpenAI Whisper and HuggingFace transcriber configurations,
        audio capture parameters, voice activity detection settings, and
        application behavior options. Automatically validates all settings
        after loading to ensure they are within acceptable ranges.

        Environment Variables:
            TRANSCRIBER_TYPE: 'openai' or 'huggingface' (default: 'openai')
            OPENAI_MODEL_SIZE: Model size for OpenAI Whisper (default: 'medium')
            WHISPER_MODEL_ID: HuggingFace model ID (default: 'openai/whisper-large-v3')
            DEVICE: Computing device (default: auto-detect)
            SAMPLE_RATE: Audio sample rate in Hz (default: 16000)
            CHANNELS: Audio channels (default: 1)
            CHUNK_DURATION: Recording chunk duration (default: 30.0)
            MAX_RECORDING_DURATION: Maximum recording time (default: 300.0)
            VAD_AGGRESSIVENESS: VAD sensitivity 0-3 (default: 2)
            SILENCE_THRESHOLD: Silence duration for auto-stop (default: 1.0)
            SHOW_AUDIO_DEVICES: Show device list on startup (default: false)
        """
        # Model configuration
        self.transcriber_type = os.getenv("TRANSCRIBER_TYPE", "openai")  # "openai" or "huggingface"
        self.model_id = os.getenv("WHISPER_MODEL_ID", "openai/whisper-large-v3")  # For HF
        self.openai_model_size = os.getenv("OPENAI_MODEL_SIZE", "medium")  # For OpenAI Whisper
        self.device = os.getenv("DEVICE", None)  # Auto-detect if None

        # Audio configuration
        self.sample_rate = int(os.getenv("SAMPLE_RATE", "16000"))
        self.channels = int(os.getenv("CHANNELS", "1"))
        self.chunk_duration = float(os.getenv("CHUNK_DURATION", "30.0"))
        self.max_recording_duration = float(os.getenv("MAX_RECORDING_DURATION", "300.0"))

        # Application configuration
        self.show_audio_devices = os.getenv("SHOW_AUDIO_DEVICES", "false").lower() == "true"

        # Voice Activity Detection configuration
        self.vad_aggressiveness = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
        self.silence_threshold = float(os.getenv("SILENCE_THRESHOLD", "1.0"))

        # Validate configuration
        self._validate()

    def _validate(self):
        """
        Validate configuration parameters and raise errors for invalid values.

        Checks all configuration parameters against acceptable ranges and formats.
        Provides warnings for unusual but valid values and raises ValueError
        exceptions for invalid configurations that would cause application failures.
        Ensures audio parameters are compatible with Whisper requirements.

        Raises:
            ValueError: If any configuration parameter is outside acceptable ranges

        Warnings:
            Prints warnings for unusual but valid sample rates or other parameters
        """
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            print(f"Warning: Unusual sample rate {self.sample_rate}Hz. Whisper works best with 16000Hz.")

        if self.channels not in [1, 2]:
            raise ValueError(f"Unsupported number of channels: {self.channels}. Use 1 (mono) or 2 (stereo).")

        if self.chunk_duration <= 0:
            raise ValueError(f"Chunk duration must be positive, got {self.chunk_duration}")

        if self.max_recording_duration <= 0:
            raise ValueError(f"Max recording duration must be positive, got {self.max_recording_duration}")

        if not 0 <= self.vad_aggressiveness <= 3:
            raise ValueError(f"VAD aggressiveness must be 0-3, got {self.vad_aggressiveness}")

        if self.silence_threshold <= 0:
            raise ValueError(f"Silence threshold must be positive, got {self.silence_threshold}")

    def print_config(self):
        """
        Display current configuration settings in a formatted output.

        Prints all active configuration parameters in a human-readable format.
        Shows different information based on the selected transcriber type.
        Useful for debugging configuration issues and verifying settings
        during application startup.
        """
        print("Configuration:")
        print(f"  Transcriber Type: {self.transcriber_type}")
        if self.transcriber_type == "openai":
            print(f"  OpenAI Model Size: {self.openai_model_size}")
        else:
            print(f"  HuggingFace Model ID: {self.model_id}")
            print(f"  Device: {self.device or 'auto-detect'}")
        print(f"  Sample Rate: {self.sample_rate}Hz")
        print(f"  Channels: {self.channels}")
        print(f"  Chunk Duration: {self.chunk_duration}s")
        print(f"  Max Recording Duration: {self.max_recording_duration}s")
        print(f"  VAD Aggressiveness: {self.vad_aggressiveness}")
        print(f"  Silence Threshold: {self.silence_threshold}s")

    @classmethod
    def create_env_file(cls, file_path: str = ".env.example"):
        """
        Create an example environment file with all available configuration options.

        Generates a template .env file containing all supported environment
        variables with their default values and explanatory comments. Users
        can copy this to .env and modify values as needed for their setup.

        Args:
            file_path (str): Path where to create the example file (default: '.env.example')

        Note:
            This is a class method, so it can be called without creating a Config instance.
            Example usage: Config.create_env_file('.env')
        """
        env_content = """# Audio Transcription Configuration

# Transcriber type: "openai" (recommended) or "huggingface"
TRANSCRIBER_TYPE=openai

# For OpenAI Whisper: tiny, base, small, medium, large
OPENAI_MODEL_SIZE=base

# For HuggingFace Whisper (only if TRANSCRIBER_TYPE=huggingface)
WHISPER_MODEL_ID=openai/whisper-large-v3
DEVICE=

# Audio configuration
SAMPLE_RATE=16000
CHANNELS=1
CHUNK_DURATION=30.0
MAX_RECORDING_DURATION=300.0

# Show available audio devices on startup
SHOW_AUDIO_DEVICES=false
"""
        with open(file_path, 'w') as f:
            f.write(env_content)
        print(f"Created example environment file: {file_path}")


# Available Whisper model options
WHISPER_MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3"
}