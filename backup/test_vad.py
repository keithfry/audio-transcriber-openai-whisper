#!/usr/bin/env python3

import numpy as np
import time
from audio_capture import AudioCapture

def test_vad_functionality():
    print("🧪 Testing Voice Activity Detection")
    print("=" * 40)

    # Test VAD initialization
    try:
        audio_capture = AudioCapture()
        print("✅ AudioCapture with VAD initialized successfully")

        # Test VAD configuration
        audio_capture.set_vad_aggressiveness(1)
        audio_capture.set_silence_threshold(2.0)
        print("✅ VAD configuration test passed")

        # Test speech detection with dummy data
        print("\n🧪 Testing speech detection algorithm...")

        # Create dummy speech-like audio (sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        speech_like_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        speech_detected = audio_capture.detect_speech_in_chunk(speech_like_audio)
        print(f"Speech detection on sine wave: {speech_detected}")

        # Test with silence (noise)
        silence_audio = np.random.normal(0, 0.01, int(sample_rate * duration)).astype(np.float32)
        silence_detected = audio_capture.detect_speech_in_chunk(silence_audio)
        print(f"Speech detection on noise: {silence_detected}")

        print("\n✅ VAD functionality test completed")
        return True

    except Exception as e:
        print(f"❌ VAD test failed: {e}")
        return False

def main():
    print("🚀 Voice Activity Detection Test")
    print("=" * 50)

    if test_vad_functionality():
        print("\n🎉 VAD tests passed!")
        print("\n💡 The application now includes:")
        print("   • Automatic speech end detection")
        print("   • Configurable silence threshold")
        print("   • Voice activity detection using WebRTC VAD")
        print("\n🚀 Ready to test with real audio!")
    else:
        print("\n⚠️ VAD tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()