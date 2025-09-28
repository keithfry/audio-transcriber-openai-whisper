import sounddevice as sd
import numpy as np
import threading
import queue
import time
import webrtcvad
from typing import Optional, Tuple, List


class AudioCapture:
    def __init__(self, sample_rate: int = 16000, channels: int = 1, chunk_duration: float = 30.0):
        """
        Initialize audio capture with voice activity detection and energy-based speech detection.

        Sets up audio recording parameters, WebRTC VAD for speech detection, and energy-based
        thresholds for robust speech/silence detection. Uses static noise floor values to
        avoid calibration delays. Configures frame sizes optimized for real-time processing.

        Args:
            sample_rate (int): Audio sampling rate in Hz (default: 16000, optimal for Whisper)
            channels (int): Number of audio channels (default: 1 for mono)
            chunk_duration (float): Maximum recording chunk length in seconds (default: 30.0)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        self.recording_thread = None

        # Voice Activity Detection
        self.vad = webrtcvad.Vad(1)  # Start with less aggressive VAD
        self.vad_frame_duration = 0.03  # 30ms frames for VAD
        self.vad_frame_size = int(sample_rate * self.vad_frame_duration)
        self.silence_threshold = 1.5  # seconds of silence before stopping
        self.min_speech_duration = 0.5  # minimum seconds of speech to consider valid

        # Energy-based detection with reasonable static values
        self.noise_floor = 0.001  # Conservative static noise floor
        self.energy_threshold = 0.01  # Higher threshold for better speech detection
        self.energy_window_size = int(sample_rate * 0.1)  # 100ms windows
        self.noise_calibrated = True  # Skip calibration
        self.min_speech_energy_ratio = 5.0  # Speech must be 5x louder than noise floor

    def list_audio_devices(self) -> None:
        """
        Display all available audio input devices on the system.

        Queries the system for available audio devices and prints their information
        including device IDs, names, and capabilities. Useful for troubleshooting
        audio input issues or selecting specific microphones.
        """
        print("Available audio devices:")
        print(sd.query_devices())

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """
        Real-time audio callback function for continuous audio stream processing.

        Called by sounddevice for each audio buffer. Converts incoming audio data
        to the correct format and queues it for VAD processing. Handles audio
        stream status monitoring and ensures thread-safe data transfer.

        Args:
            indata (np.ndarray): Raw audio data from the microphone
            frames (int): Number of audio frames in the buffer
            time_info: Timing information from sounddevice (unused)
            status: Stream status flags for error detection
        """
        _ = frames, time_info  # Mark parameters as intentionally unused
        if status:
            print(f"Audio callback status: {status}")

        # Convert to mono if needed and preserve data type
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1, dtype=np.float32)
        else:
            audio_data = indata.flatten().astype(np.float32)

        # Add to queue without additional processing
        self.audio_queue.put(audio_data.copy())

    def start_recording(self, device: Optional[int] = None) -> None:
        """
        Start continuous audio recording stream from the specified device.

        Initializes a sounddevice input stream with the configured audio parameters.
        The stream runs continuously in the background, calling the audio_callback
        for each buffer of audio data. Sets up proper error handling and ensures
        only one recording session can be active at a time.

        Args:
            device (Optional[int]): Audio input device ID. If None, uses system default.
        """
        if self.is_recording:
            print("Already recording!")
            return

        self.is_recording = True

        try:
            self.stream = sd.InputStream(
                device=device,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=1024,
                dtype='float32'
            )
            self.stream.start()
            print(f"Recording started at {self.sample_rate}Hz...")
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            raise

    def stop_recording(self) -> None:
        """
        Stop the active audio recording stream and clean up resources.

        Safely terminates the sounddevice input stream, sets recording flags to false,
        and ensures proper cleanup of audio resources. Can be called multiple times
        safely. Used both for normal recording completion and emergency shutdown.
        """
        if not self.is_recording:
            return

        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        print("Recording stopped.")

    # Noise floor calibration removed - using static value for immediate start

    def record_with_vad(self, max_duration: float = 30.0, min_duration: float = 0.5) -> np.ndarray:
        """
        Record audio with voice activity detection to automatically stop when speech ends.

        Core recording method that uses both WebRTC VAD and energy-based detection
        to identify speech and silence periods. Records until speech stops for the
        configured silence threshold duration, or until max_duration is reached.
        Ensures minimum recording duration to avoid capturing just noise clicks.

        The method continuously monitors audio for:
        1. Speech activity using WebRTC VAD (optimized for voice)
        2. Energy levels compared to noise floor
        3. Silence duration tracking for auto-stop
        4. Minimum speech duration validation

        Args:
            max_duration (float): Maximum recording time in seconds
            min_duration (float): Minimum recording time in seconds

        Returns:
            np.ndarray: Recorded audio data as float32 array, or empty array if no speech
        """
        """Record audio with voice activity detection."""

        print("🎙️ Recording... Speak now! (Press Ctrl+C to stop manually)")
        print("   Dots will appear while speech is detected...")

        # Clear the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        self.start_recording()

        audio_data = []
        start_time = time.time()
        last_speech_time = start_time
        speech_detected = False

        try:
            while time.time() - start_time < max_duration:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    audio_data.append(chunk)

                    # Check for voice activity in recent audio
                    if len(audio_data) > 0:
                        recent_audio = np.concatenate(audio_data[-5:])  # Last ~0.5 second

                        # Use both VAD and energy detection
                        vad_speech = self.detect_speech_in_chunk(recent_audio)
                        energy_speech = self.detect_speech_by_energy(recent_audio)

                        # Require BOTH methods to agree for more reliable detection
                        if vad_speech and energy_speech:
                            last_speech_time = time.time()
                            speech_detected = True
                            print(".", end="", flush=True)  # Show activity

                        # Stop if we've had enough silence after detecting speech
                        silence_duration = time.time() - last_speech_time
                        total_duration = time.time() - start_time

                        # More aggressive stopping - shorter silence threshold after speech
                        effective_threshold = self.silence_threshold if speech_detected else 3.0

                        if (speech_detected and
                            silence_duration > effective_threshold and
                            total_duration > min_duration):
                            print(f"\n🔇 Silence detected for {silence_duration:.1f}s - stopping recording")
                            break
                        elif total_duration > max_duration * 0.8:  # Warn near max duration
                            print(f"\n⏰ Recording for {total_duration:.1f}s - will stop at {max_duration}s")

                except queue.Empty:
                    continue

        except KeyboardInterrupt:
            print("\nRecording interrupted by user.")
        finally:
            self.stop_recording()

        if audio_data:
            result = np.concatenate(audio_data)
            duration = len(result) / self.sample_rate
            print(f"📊 Recorded {duration:.2f} seconds of audio")
            if not speech_detected:
                print("⚠️ No speech detected in recording")
            return result
        else:
            print("No audio data recorded")
            return np.array([])

    def detect_speech_in_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect speech in an audio chunk using WebRTC VAD with frame-by-frame analysis.

        Processes audio in 30ms frames (optimal for WebRTC VAD) and determines if
        the chunk contains speech. Converts float32 audio to int16 format required
        by WebRTC VAD, then analyzes each frame. Returns True if any frame in the
        chunk is classified as containing speech.

        Args:
            audio_chunk (np.ndarray): Audio data to analyze (float32 format)

        Returns:
            bool: True if speech is detected in the chunk, False otherwise
        """
        """Detect speech in audio chunk using WebRTC VAD."""
        if len(audio_chunk) < self.vad_frame_size:
            return False

        # Ensure audio is properly normalized before VAD conversion
        max_val = np.max(np.abs(audio_chunk))
        if max_val > 0:
            normalized_audio = audio_chunk / max_val
        else:
            normalized_audio = audio_chunk

        # Convert to 16-bit PCM format required by WebRTC VAD
        # Clamp values to prevent overflow
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        audio_int16 = (normalized_audio * 32767).astype(np.int16)

        # Process in 30ms frames
        speech_frames = 0
        total_frames = 0

        for i in range(0, len(audio_int16) - self.vad_frame_size, self.vad_frame_size):
            frame = audio_int16[i:i + self.vad_frame_size].tobytes()

            try:
                if self.vad.is_speech(frame, self.sample_rate):
                    speech_frames += 1
                total_frames += 1
            except Exception as e:
                # VAD can fail on some audio, skip this frame
                continue

        # More strict - require more frames to contain speech
        if total_frames > 0:
            speech_ratio = speech_frames / total_frames
            return speech_ratio > 0.4  # Increased from 0.15 to 0.4
        return False

    def detect_speech_by_energy(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect speech using energy-based analysis with noise floor comparison.

        Analyzes audio energy levels by computing RMS (root mean square) values
        and comparing against the established noise floor. Speech is detected when
        energy exceeds both the absolute energy threshold and the noise floor ratio.
        This provides a secondary speech detection method that complements WebRTC VAD.

        Args:
            audio_chunk (np.ndarray): Audio data to analyze

        Returns:
            bool: True if energy levels indicate speech, False for silence/noise
        """
        """Detect speech using energy-based method as backup."""
        if len(audio_chunk) < self.energy_window_size:
            return False

        # Calculate RMS energy in overlapping windows
        speech_windows = 0
        total_windows = 0
        peak_energy = 0

        for i in range(0, len(audio_chunk) - self.energy_window_size, self.energy_window_size // 2):
            window = audio_chunk[i:i + self.energy_window_size]
            rms_energy = np.sqrt(np.mean(window**2))
            peak_energy = max(peak_energy, rms_energy)

            # Use adaptive threshold based on noise floor
            effective_threshold = self.energy_threshold
            if self.noise_calibrated:
                effective_threshold = max(self.energy_threshold, self.noise_floor * self.min_speech_energy_ratio)

            if rms_energy > effective_threshold:
                speech_windows += 1
            total_windows += 1

        # More strict requirements for energy-based detection
        if total_windows > 0:
            energy_ratio = speech_windows / total_windows
            # Also require peak energy to be significantly above noise floor
            peak_above_noise = not self.noise_calibrated or peak_energy > (self.noise_floor * 2.0)
            return energy_ratio > 0.5 and peak_above_noise  # Increased from 0.3 to 0.5
        return False

    def record_chunk(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Record a single audio chunk for the specified duration.

        Records audio for a fixed duration without voice activity detection.
        Useful for testing, calibration, or when VAD is not desired. Collects
        audio data from the stream and returns it as a single array.

        Args:
            duration (Optional[float]): Recording duration in seconds.
                                       If None, uses self.chunk_duration.

        Returns:
            np.ndarray: Recorded audio data for the specified duration
        """
        if duration is None:
            duration = self.chunk_duration

        print(f"Recording for {duration} seconds... (Press Ctrl+C to stop early)")

        # Clear the queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        self.start_recording()

        audio_data = []
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    audio_data.append(chunk)
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            print("\nRecording interrupted by user.")
        finally:
            self.stop_recording()

        if audio_data:
            result = np.concatenate(audio_data)
            print(f"Recorded {len(result) / self.sample_rate:.2f} seconds of audio")
            return result
        else:
            print("No audio data recorded")
            return np.array([])

    def record_with_chunking(self, max_duration: float = 300.0) -> List[np.ndarray]:
        """
        Record long audio sessions split into multiple chunks for processing.

        Records audio in fixed-duration chunks, useful for very long recordings
        that need to be processed incrementally. Each chunk can be transcribed
        separately to handle memory constraints and provide incremental results.

        Args:
            max_duration (float): Total maximum recording duration in seconds

        Returns:
            List[np.ndarray]: List of audio chunks, each of chunk_duration length
        """
        print(f"Recording with automatic chunking every {self.chunk_duration} seconds...")
        print("Press Ctrl+C to stop recording")

        chunks = []
        start_time = time.time()

        try:
            while time.time() - start_time < max_duration:
                chunk = self.record_chunk(self.chunk_duration)
                if len(chunk) > 0:
                    chunks.append(chunk)
                else:
                    break
        except KeyboardInterrupt:
            print("\nRecording session ended by user.")

        print(f"Recorded {len(chunks)} chunks totaling {sum(len(chunk) for chunk in chunks) / self.sample_rate:.2f} seconds")
        return chunks

    def get_audio_level(self, audio_data: np.ndarray) -> float:
        """
        Calculate the RMS (root mean square) audio level for volume monitoring.

        Computes the RMS level of audio data, which represents the average
        energy/volume. Used for energy-based speech detection and audio
        level monitoring during recording.

        Args:
            audio_data (np.ndarray): Audio data to analyze

        Returns:
            float: RMS level of the audio (0.0 to 1.0 range)
        """
        if len(audio_data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_data**2)))

    def set_vad_aggressiveness(self, level: int):
        """
        Configure WebRTC VAD aggressiveness level for speech detection sensitivity.

        Sets how aggressively the VAD filters out non-speech. Higher levels are
        more aggressive at filtering but may miss quiet speech. Lower levels are
        more sensitive but may trigger on background noise.

        Args:
            level (int): VAD aggressiveness (0=least aggressive, 3=most aggressive)
        """
        """Set VAD aggressiveness level (0-3, higher = more aggressive)."""
        if 0 <= level <= 3:
            self.vad = webrtcvad.Vad(level)  # Recreate VAD with new aggressiveness
            print(f"VAD aggressiveness set to {level}")
        else:
            print("VAD aggressiveness must be between 0 and 3")

    def set_silence_threshold(self, seconds: float):
        """
        Set the duration of silence required before stopping recording.

        Configures how long the system waits after detecting silence before
        automatically stopping the recording. Longer thresholds allow for
        natural pauses in speech, shorter thresholds provide quicker response.

        Args:
            seconds (float): Silence duration threshold in seconds
        """
        """Set how many seconds of silence before stopping recording."""
        self.silence_threshold = max(0.1, seconds)
        print(f"Silence threshold set to {self.silence_threshold}s")

    def set_energy_threshold(self, threshold: float):
        """
        Set the energy threshold for speech detection.

        Configures the minimum energy level required to classify audio as speech.
        Higher thresholds require louder speech but reduce false positives from
        background noise. Lower thresholds are more sensitive to quiet speech.

        Args:
            threshold (float): Energy threshold level (typically 0.001 to 0.1)
        """
        """Set energy threshold for speech detection."""
        self.energy_threshold = max(0.001, threshold)
        print(f"Energy threshold set to {self.energy_threshold}")

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess recorded audio data for optimal transcription quality.

        Applies audio processing steps to improve transcription accuracy:
        1. Normalizes audio levels to optimal range
        2. Applies gentle noise reduction
        3. Ensures proper amplitude scaling
        4. Maintains audio quality while reducing artifacts

        Args:
            audio_data (np.ndarray): Raw recorded audio data

        Returns:
            np.ndarray: Processed audio data ready for transcription
        """
        if len(audio_data) == 0:
            return audio_data

        # Minimal preprocessing to preserve speech quality
        # Remove DC offset first
        audio_data = audio_data - np.mean(audio_data)

        # Light normalization - only if signal is very weak or very strong
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:  # Only normalize if clipping would occur
            audio_data = audio_data / max_val
        elif max_val < 0.01:  # Only amplify if signal is very weak
            audio_data = audio_data / max_val * 0.1

        return audio_data

    def __enter__(self):
        """
        Context manager entry point for automatic resource management.

        Enables using AudioCapture with 'with' statements for automatic
        cleanup of audio resources. Returns self to allow method chaining.

        Returns:
            AudioCapture: Self for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point ensuring proper resource cleanup.

        Automatically stops recording and cleans up resources when exiting
        a 'with' statement, regardless of whether an exception occurred.
        Ensures audio streams are properly closed.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.stop_recording()