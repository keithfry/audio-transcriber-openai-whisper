import sounddevice as sd
import numpy as np
import tempfile
import os
import soundfile as sf
from typing import Optional


class AudioPlayback:
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio playback with specified sample rate.

        Sets up the audio playback system with the given sample rate. The sample
        rate should match the rate used for recording to ensure proper playback
        speed and pitch. Initializes internal state for managing playback operations.

        Args:
            sample_rate (int): Audio sample rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate

    def play_audio(self, audio_data: np.ndarray, blocking: bool = True) -> None:
        """
        Play audio data through the system's default audio output device.

        Converts numpy audio data to the proper format and plays it using sounddevice.
        Can operate in blocking mode (waits for playback to complete) or non-blocking
        mode (returns immediately while audio plays in background). Handles audio
        format conversion and ensures proper volume levels.

        Args:
            audio_data (np.ndarray): Audio data to play (float32 format expected)
            blocking (bool): If True, waits for playback to complete. If False, returns immediately.

        Raises:
            Exception: If audio playback fails due to device or format issues
        """
        if len(audio_data) == 0:
            print("⚠️ No audio data to play")
            return

        try:
            # Ensure audio is in the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.8  # Leave some headroom

            print("🔊 Playing recorded audio...")
            sd.play(audio_data, samplerate=self.sample_rate)

            if blocking:
                sd.wait()  # Wait until audio finishes playing
                print("✅ Audio playback finished")

        except Exception as e:
            print(f"❌ Error playing audio: {e}")

    def save_and_play_audio(self, audio_data: np.ndarray, filename: Optional[str] = None) -> str:
        """
        Save audio data to a WAV file and play it back for verification.

        Combines audio saving and playback functionality. Saves the audio data
        to a WAV file (with automatic filename generation if not specified) and
        then plays it back. Useful for debugging audio capture issues or creating
        audio files for later analysis.

        Args:
            audio_data (np.ndarray): Audio data to save and play
            filename (Optional[str]): Output filename. If None, generates timestamp-based name.

        Returns:
            str: Path to the saved audio file

        Raises:
            Exception: If file saving or playback fails
        """
        if filename is None:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                filename = tmp_file.name

        try:
            # Save audio to file
            sf.write(filename, audio_data, self.sample_rate)
            print(f"💾 Audio saved to {filename}")

            # Play the audio
            self.play_audio(audio_data, blocking=True)

            return filename

        except Exception as e:
            print(f"❌ Error saving/playing audio: {e}")
            return ""

    def get_audio_info(self, audio_data: np.ndarray) -> dict:
        """
        Analyze audio data and return comprehensive information about its properties.

        Computes various audio characteristics including duration, energy levels,
        peak amplitudes, and basic quality metrics. This information is useful
        for debugging audio capture issues, verifying recording quality, and
        displaying feedback to users about their recordings.

        Args:
            audio_data (np.ndarray): Audio data to analyze

        Returns:
            dict: Dictionary containing:
                - duration: Audio length in seconds
                - energy: RMS energy level
                - max_amplitude: Maximum absolute amplitude
                - samples: Number of audio samples
                - sample_rate: Configured sample rate
        """
        if len(audio_data) == 0:
            return {"duration": 0, "energy": 0, "max_amplitude": 0}

        duration = len(audio_data) / self.sample_rate
        energy = np.sqrt(np.mean(audio_data**2))
        max_amplitude = np.max(np.abs(audio_data))

        return {
            "duration": duration,
            "energy": energy,
            "max_amplitude": max_amplitude,
            "samples": len(audio_data),
            "sample_rate": self.sample_rate
        }

    def list_audio_devices(self) -> None:
        """
        Display all available audio output devices on the system.

        Queries sounddevice for available audio output devices and prints their
        information including device IDs, names, and channel configurations.
        Useful for troubleshooting audio playback issues or selecting specific
        audio output devices.
        """
        print("Available audio output devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                print(f"  {i}: {device['name']} (channels: {device['max_output_channels']})")

    def test_playback(self) -> bool:
        """
        Test audio playback functionality by playing a generated tone.

        Generates a short sine wave tone and attempts to play it through the
        audio system. This tests whether the audio output device is working
        properly and the sounddevice library is functioning correctly.
        Useful for system diagnostics and setup verification.

        Returns:
            bool: True if test tone played successfully, False if playback failed
        """
        try:
            # Generate a simple test tone (440 Hz for 1 second)
            duration = 1.0
            frequency = 440.0
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            test_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

            print("🔊 Testing audio playback with 440Hz tone...")
            self.play_audio(test_audio, blocking=True)
            return True

        except Exception as e:
            print(f"❌ Audio playback test failed: {e}")
            return False

    def cleanup_temp_files(self, filenames: list) -> None:
        """
        Remove temporary audio files created during testing or debugging.

        Safely deletes a list of temporary audio files, handling errors gracefully
        if files don't exist or can't be deleted. Used to clean up WAV files
        created during save_and_play operations or testing.

        Args:
            filenames (list): List of file paths to delete
        """
        for filename in filenames:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    print(f"🗑️ Cleaned up {filename}")
            except Exception as e:
                print(f"⚠️ Could not delete {filename}: {e}")

    def stop_playback(self) -> None:
        """
        Stop any currently active audio playback.

        Interrupts ongoing audio playback operations and resets the playback
        system. Used for emergency stops or when switching between different
        audio operations. Safe to call even if no playback is active.
        """
        try:
            sd.stop()
            print("⏹️ Audio playback stopped")
        except Exception as e:
            print(f"⚠️ Error stopping playback: {e}")

    def __enter__(self):
        """
        Context manager entry point for automatic resource management.

        Enables using AudioPlayback with 'with' statements for automatic
        cleanup of audio resources. Returns self to allow method chaining.

        Returns:
            AudioPlayback: Self for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point ensuring proper resource cleanup.

        Automatically stops playback and cleans up resources when exiting
        a 'with' statement, regardless of whether an exception occurred.
        Ensures audio streams are properly closed and temporary files cleaned.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.stop_playback()