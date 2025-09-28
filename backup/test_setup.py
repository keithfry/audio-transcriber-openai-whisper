#!/usr/bin/env python3

import numpy as np
import sys
from config import Config
from audio_capture import AudioCapture
from transcriber import WhisperTranscriber

def test_configuration():
    print("🧪 Testing configuration...")
    try:
        config = Config()
        config.print_config()
        print("✅ Configuration test passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_audio_capture():
    print("\n🧪 Testing audio capture initialization...")
    try:
        audio_capture = AudioCapture()
        print("✅ Audio capture initialization passed")

        # Test audio preprocessing with dummy data
        dummy_audio = np.random.randn(1000).astype(np.float32)
        processed = audio_capture.preprocess_audio(dummy_audio)
        print(f"✅ Audio preprocessing test passed (processed {len(processed)} samples)")
        return True
    except Exception as e:
        print(f"❌ Audio capture test failed: {e}")
        return False

def test_transcriber_loading():
    print("\n🧪 Testing Whisper model loading...")
    try:
        # Use a smaller model for testing
        transcriber = WhisperTranscriber(model_id="openai/whisper-tiny")
        model_info = transcriber.get_model_info()
        print(f"✅ Model loaded: {model_info['model_id']} on {model_info['device']}")

        # Test with dummy audio
        print("🧪 Testing transcription with dummy audio...")
        dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
        result = transcriber.transcribe_audio(dummy_audio)
        print(f"✅ Transcription test completed (output: '{result['text'][:50]}...')")
        return True
    except Exception as e:
        print(f"❌ Transcriber test failed: {e}")
        return False

def main():
    print("🚀 Audio Transcription System - Setup Test")
    print("=" * 50)

    tests = [
        ("Configuration", test_configuration),
        ("Audio Capture", test_audio_capture),
        ("Whisper Model", test_transcriber_loading),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        if test_func():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        print("\n🚀 To start the application, run:")
        print("   source .env/bin/activate && python main.py")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()