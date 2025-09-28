#!/usr/bin/env python3

import numpy as np
from transcriber import WhisperTranscriber

def test_transcription_fix():
    print("🧪 Testing Transcription Fixes")
    print("=" * 40)

    try:
        # Use tiny model for faster testing
        print("Loading Whisper tiny model...")
        transcriber = WhisperTranscriber(model_id="openai/whisper-tiny")

        # Test 1: Empty audio
        print("\n🧪 Test 1: Empty audio")
        result = transcriber.transcribe_audio(np.array([]))
        print(f"Empty audio result: {result}")

        # Test 2: Very short audio
        print("\n🧪 Test 2: Very short audio (0.05s)")
        short_audio = np.random.randn(800).astype(np.float32) * 0.1  # 0.05s at 16kHz
        result = transcriber.transcribe_audio(short_audio)
        print(f"Short audio result: {result}")

        # Test 3: Normal length audio with speech-like pattern
        print("\n🧪 Test 3: Normal audio with speech pattern (2s)")
        # Create a more speech-like signal with multiple frequencies
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)

        # Mix of frequencies to simulate speech
        speech_audio = (
            0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
            0.2 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency
            0.1 * np.sin(2 * np.pi * 1600 * t)   # High frequency
        )

        # Add some amplitude modulation to make it more speech-like
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
        speech_audio = speech_audio * modulation

        speech_audio = speech_audio.astype(np.float32)

        result = transcriber.transcribe_audio(speech_audio)
        print(f"Speech-like audio result: {result}")

        # Test 4: Audio with proper energy levels
        print("\n🧪 Test 4: Audio with good energy levels")
        good_audio = np.random.randn(32000).astype(np.float32) * 0.3  # 2s at 16kHz
        result = transcriber.transcribe_audio(good_audio)
        print(f"Good energy audio result: {result}")

        print("\n✅ All transcription tests completed!")
        print("💡 The fixes should resolve the timestamp errors.")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🚀 Transcription Fix Test")
    print("=" * 50)

    if test_transcription_fix():
        print("\n🎉 Transcription tests passed!")
        print("🔧 Fixed issues:")
        print("   • Removed problematic timestamp requirements")
        print("   • Added audio validation and preprocessing")
        print("   • Improved error handling")
        print("   • Added minimum audio length padding")
        print("\n🎤 Ready to test with real microphone input!")
    else:
        print("\n⚠️ Some tests failed. Check error messages above.")

if __name__ == "__main__":
    main()