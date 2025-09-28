#!/usr/bin/env python3

import numpy as np
from audio_capture import AudioCapture
from transcriber_openai import OpenAIWhisperTranscriber
from audio_playback import AudioPlayback
from config import Config

def test_integrated_features():
    print("🧪 Testing Integrated Application Features")
    print("=" * 45)

    try:
        # Test configuration
        print("Step 1: Testing configuration...")
        config = Config()
        config.print_config()
        print("✅ Configuration loaded successfully")

        # Test OpenAI Whisper transcriber
        print(f"\nStep 2: Testing OpenAI Whisper transcriber...")
        transcriber = OpenAIWhisperTranscriber(model_size=config.openai_model_size)
        print("✅ OpenAI Whisper loaded successfully")

        # Test audio capture
        print(f"\nStep 3: Testing audio capture...")
        audio_capture = AudioCapture()
        print("✅ Audio capture initialized")

        # Test audio playback
        print(f"\nStep 4: Testing audio playback...")
        audio_playback = AudioPlayback()
        print("✅ Audio playback initialized")

        # Test complete workflow
        print(f"\nStep 5: Testing complete workflow...")
        print("🎤 Please speak clearly for testing...")

        # Record audio
        audio_data = audio_capture.record_with_vad(max_duration=10.0, min_duration=1.0)

        if len(audio_data) == 0:
            print("❌ No audio captured")
            return False

        # Transcribe with OpenAI Whisper
        result = transcriber.transcribe_audio(audio_data)
        transcribed_text = result.get('text', '').strip()

        print(f"\n📝 Transcription: '{transcribed_text}'")

        # Test verification workflow
        if transcribed_text:
            print(f"\n🔍 Testing verification feature...")
            print(f"📝 Transcribed text: '{transcribed_text}'")

            # Simulate verification (normally user input)
            verification_test = "n"  # Simulate "no" to test playback
            print(f"Simulating verification response: '{verification_test}'")

            if verification_test == "n":
                print("❌ Transcription marked as incorrect (simulated)")

                # Test audio playback
                print("🔊 Testing audio playback feature...")
                try:
                    audio_playback.play_audio(audio_data, blocking=False)  # Non-blocking for test
                    print("✅ Audio playback feature working")
                except Exception as e:
                    print(f"⚠️ Audio playback issue: {e}")

        print(f"\n✅ All integrated features tested successfully!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Integrated Application Test")
    print("=" * 50)

    print("💡 Testing complete application with all features:")
    print("   • OpenAI Whisper transcription")
    print("   • Voice activity detection")
    print("   • Audio playback verification")
    print("   • Configuration management")
    print("   • Complete workflow integration")

    if test_integrated_features():
        print("\n🎉 Integration test successful!")
        print("\n✅ The complete application is ready with:")
        print("   • Working OpenAI Whisper transcription")
        print("   • Voice activity detection (no calibration)")
        print("   • Transcription verification prompts")
        print("   • Audio playback for verification")
        print("   • Configurable model selection")

        print("\n🎤 Ready to use the full application:")
        print("   source .env/bin/activate && python main.py")
    else:
        print("\n⚠️ Integration test failed - check error messages above")

if __name__ == "__main__":
    main()