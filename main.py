#!/usr/bin/env python3

import sys
import signal
import time
from typing import Optional
import numpy as np

from audio_capture import AudioCapture
from transcriber_openai import OpenAIWhisperTranscriber
from audio_playback import AudioPlayback
from config import Config


class AudioTranscriptionApp:
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the audio transcription application with configuration and signal handlers.

        Sets up the main application state, loads configuration, and registers signal handlers
        for graceful shutdown on SIGINT (Ctrl+C) and SIGTERM. Initializes component placeholders
        that will be populated during the initialize_components() call.

        Args:
            config (Optional[Config]): Configuration object. If None, creates default Config.
        """
        self.config = config or Config()
        self.running = True
        self.transcriber = None
        self.audio_capture = None
        self.audio_playback = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Handle interrupt signals for graceful application shutdown.

        Called when the application receives SIGINT (Ctrl+C) or SIGTERM signals.
        Sets the running flag to False, which triggers cleanup in the main loop and
        allows components to shut down properly.

        Args:
            signum (int): Signal number that was received
            frame: Current stack frame (unused)
        """
        _ = signum, frame  # Mark parameters as intentionally unused
        print("\n\nReceived interrupt signal. Shutting down gracefully...")
        self.running = False

    def initialize_components(self):
        """
        Initialize all application components: transcriber, audio capture, and playback.

        Loads the OpenAI Whisper transcriber with the configured model size, initializes
        audio capture with voice activity detection settings, and sets up audio playback
        for verification. Displays component information and optionally lists available
        audio devices. Exits the application if any component fails to initialize.

        Raises:
            SystemExit: If any component initialization fails
        """
        print("🎙️ Audio Transcription System")
        print("=" * 40)

        try:
            # Initialize transcriber based on configuration
            print(f"\n📥 Loading {self.config.transcriber_type} Whisper model...")

            if self.config.transcriber_type == "openai":
                self.transcriber = OpenAIWhisperTranscriber(
                    model_size=self.config.openai_model_size
                )
                model_info = self.transcriber.get_model_info()
                print(f"✅ Model Type: {model_info['model_type']}")
                print(f"✅ Model Size: {model_info['model_size']}")
                print(f"✅ Sample Rate: {model_info['sample_rate']}Hz")
            else:
                print("❌ HuggingFace transcriber not available (moved to backup)")
                print("💡 Please set TRANSCRIBER_TYPE=openai in your environment")
                sys.exit(1)

            # Initialize audio capture
            self.audio_capture = AudioCapture(
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                chunk_duration=self.config.chunk_duration
            )

            # Configure VAD settings
            self.audio_capture.set_vad_aggressiveness(self.config.vad_aggressiveness)
            self.audio_capture.set_silence_threshold(self.config.silence_threshold)

            print(f"✅ Audio capture initialized ({self.config.sample_rate}Hz, {self.config.channels} channel(s))")

            # Initialize audio playback
            self.audio_playback = AudioPlayback(sample_rate=self.config.sample_rate)
            print("✅ Audio playback initialized")

            if self.config.show_audio_devices:
                print("\n🔍 Available audio devices:")
                self.audio_capture.list_audio_devices()

        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            sys.exit(1)

    def get_user_confirmation(self, prompt: str = "Continue transcribing? (y/n): ") -> bool:
        """
        Get yes/no confirmation from the user with input validation.

        Prompts the user with a customizable message and validates their response.
        Accepts various forms of yes (y, yes, 1, true) and no (n, no, 0, false) answers.
        Continues prompting until a valid response is received or the user interrupts.

        Args:
            prompt (str): The prompt message to display to the user

        Returns:
            bool: True for yes responses, False for no responses or interruption
        """
        while self.running:
            try:
                response = input(f"\n{prompt}").strip().lower()
                if response in ['y', 'yes', '1', 'true']:
                    return True
                elif response in ['n', 'no', '0', 'false']:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                return False

    def verify_transcription(self, transcribed_text: str, audio_data: np.ndarray) -> bool:
        """
        Ask user to verify transcription accuracy and optionally play back the recorded audio.

        Displays the transcribed text and prompts the user to confirm its accuracy.
        If the user indicates the transcription is incorrect, offers to play back the
        original recorded audio so they can hear what was actually captured. This helps
        identify whether issues are with audio capture or transcription processing.

        Args:
            transcribed_text (str): The text produced by the transcription process
            audio_data (np.ndarray): The original recorded audio data for playback

        Returns:
            bool: True if user confirms transcription is correct, False otherwise
        """
        """Ask user to verify transcription accuracy and play audio if incorrect."""
        print(f"\n🔍 Please verify the transcription:")
        print(f"📝 Transcribed text: '{transcribed_text}'")

        while self.running:
            try:
                response = input("\n❓ Is this transcription correct? (y/n): ").strip().lower()
                if response in ['y', 'yes', '1', 'true']:
                    print("✅ Transcription verified as correct")
                    return True
                elif response in ['n', 'no', '0', 'false']:
                    print("❌ Transcription marked as incorrect")

                    # Get audio info
                    audio_info = self.audio_playback.get_audio_info(audio_data)
                    print(f"🎵 Audio info: {audio_info['duration']:.2f}s, "
                          f"energy: {audio_info['energy']:.4f}")

                    # Ask if they want to hear the recording
                    play_response = input("🔊 Would you like to hear what was recorded? (y/n): ").strip().lower()
                    if play_response in ['y', 'yes', '1', 'true']:
                        try:
                            print("🔊 Playing back recorded audio...")
                            self.audio_playback.play_audio(audio_data, blocking=True)
                        except Exception as e:
                            print(f"❌ Error playing audio: {e}")
                    else:
                        print("ℹ️ Audio playback skipped")

                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                return False

    def transcribe_session(self):
        """
        Main transcription loop handling recording, transcription, and verification.

        Runs continuous transcription sessions where each session:
        1. Records audio using voice activity detection (auto-stops when speech ends)
        2. Transcribes the audio using OpenAI Whisper
        3. Displays the transcription results with timing information
        4. Asks user to verify transcription accuracy
        5. Offers audio playback if transcription was incorrect
        6. Prompts user whether to continue with another session

        Handles errors gracefully and provides feedback for improving recording conditions.
        Continues until user chooses to exit or application is interrupted.
        """
        print(f"\n🎤 Starting transcription session...")
        print("🎯 Voice activity detection enabled - recording will auto-stop when you finish speaking")
        print("💡 Press Ctrl+C during recording to stop early, or answer 'no' to exit\n")

        session_count = 0

        while self.running:
            session_count += 1
            print(f"\n--- Session {session_count} ---")

            try:
                # Record audio with voice activity detection
                audio_data = self.audio_capture.record_with_vad(
                    max_duration=self.config.max_recording_duration,
                    min_duration=0.5
                )

                if len(audio_data) == 0:
                    print("⚠️ No audio recorded. Please check your microphone.")
                    continue

                # Determine transcription method based on audio length and transcriber type
                audio_duration = len(audio_data) / self.config.sample_rate

                print(f"\n🤖 Transcribing {audio_duration:.1f}s of audio...")

                # OpenAI Whisper handles long audio automatically
                result = self.transcriber.transcribe_audio(audio_data)

                # Display results
                transcribed_text = result.get("text", "").strip()

                if transcribed_text:
                    print("\n" + "="*50)
                    print("📝 TRANSCRIPTION:")
                    print("="*50)
                    print(f"{transcribed_text}")
                    print("="*50)

                    # Show timing info if available
                    if "total_duration" in result:
                        print(f"⏱️ Audio duration: {result['total_duration']:.1f}s")
                    if "num_chunks" in result:
                        print(f"📊 Processed {result['num_chunks']} chunks")

                    # Verify transcription with user
                    transcription_correct = self.verify_transcription(transcribed_text, audio_data)

                    if transcription_correct:
                        print("✅ Transcription session completed successfully")
                    else:
                        print("⚠️ Transcription was marked as incorrect")
                        print("💡 Consider:")
                        print("   • Speaking more clearly")
                        print("   • Getting closer to the microphone")
                        print("   • Reducing background noise")
                        print("   • Speaking louder")

                else:
                    print("🔇 No speech detected in the audio. Please try speaking louder or closer to the microphone.")

            except KeyboardInterrupt:
                print("\n⏹️ Recording session interrupted.")
                break
            except Exception as e:
                print(f"❌ Error during transcription: {e}")

            # Ask user if they want to continue
            if not self.get_user_confirmation():
                print("👋 Exiting transcription session...")
                break

        print(f"\n📊 Session completed. Total recordings: {session_count}")

    def run(self):
        """
        Main application entry point that orchestrates the complete workflow.

        Coordinates the entire application lifecycle:
        1. Initializes all components (transcriber, audio capture, playback)
        2. Starts the transcription session loop
        3. Handles keyboard interrupts and unexpected errors gracefully
        4. Ensures proper cleanup of resources regardless of how the app exits

        This is the primary method called by main() to run the application.
        """
        try:
            self.initialize_components()
            self.transcribe_session()
        except KeyboardInterrupt:
            print("\n⏹️ Application interrupted by user.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up application resources and stop all active components.

        Safely shuts down audio capture and playback components to release
        system resources like microphone access and audio devices. Called
        automatically when the application exits, whether normally or due to
        interruption/error. Ensures no resources are left in an inconsistent state.
        """
        print("\n🧹 Cleaning up...")
        if self.audio_capture:
            self.audio_capture.stop_recording()
        if self.audio_playback:
            self.audio_playback.stop_playback()
        print("✅ Cleanup completed.")


def main():
    """
    Application entry point that loads configuration and starts the audio transcription app.

    Handles command-line arguments (--show-devices), loads the configuration from
    environment variables, creates the main application instance, and starts the
    transcription workflow. Exits with error code 1 if configuration loading fails.
    """
    print("🚀 Starting Audio Transcription Application")

    # Load configuration
    try:
        config = Config()
        if len(sys.argv) > 1 and sys.argv[1] == "--show-devices":
            config.show_audio_devices = True
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    # Run application
    app = AudioTranscriptionApp(config)
    app.run()


if __name__ == "__main__":
    main()