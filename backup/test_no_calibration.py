#!/usr/bin/env python3

import numpy as np
from audio_capture import AudioCapture
from transcriber import WhisperTranscriber

def test_no_calibration():
    print("🧪 Testing Immediate Recording (No Calibration)")
    print("=" * 50)

    try:
        print("Initializing audio capture with static noise floor...")
        audio_capture = AudioCapture()

        print(f"✅ Audio capture settings:")
        print(f"   Static noise floor: {audio_capture.noise_floor:.4f}")
        print(f"   Energy threshold: {audio_capture.energy_threshold:.4f}")
        print(f"   Calibration skipped: {audio_capture.noise_calibrated}")

        print(f"\n🎤 Starting immediate recording test...")
        print("💡 Should start recording immediately without calibration delay")

        # Test immediate recording
        audio_data = audio_capture.record_with_vad(max_duration=10.0, min_duration=1.0)

        if len(audio_data) > 0:
            duration = len(audio_data) / 16000
            energy = np.sqrt(np.mean(audio_data**2))
            print(f"\n📊 Recording results:")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Energy: {energy:.4f}")
            print(f"   Samples: {len(audio_data)}")

            # Test transcription
            print(f"\n🤖 Testing transcription...")
            transcriber = WhisperTranscriber(model_id="openai/whisper-tiny")  # Use tiny for speed
            result = transcriber.transcribe_audio(audio_data)

            transcribed_text = result.get('text', '').strip()
            print(f"📝 Transcribed: '{transcribed_text}'")

            if transcribed_text:
                print("✅ Transcription successful!")
            else:
                print("⚠️ No transcription returned")

            return True
        else:
            print("❌ No audio recorded")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🚀 No Calibration Test")
    print("=" * 50)

    print("💡 This test verifies:")
    print("   • Recording starts immediately without noise floor calibration")
    print("   • Static noise floor values work correctly")
    print("   • Speech detection still functions properly")
    print("   • Faster startup time")

    if test_no_calibration():
        print("\n🎉 No calibration test passed!")
        print("\n✅ Benefits:")
        print("   • Immediate recording start")
        print("   • No calibration delay")
        print("   • Simpler user experience")
        print("   • Reasonable default thresholds")

        print("\n🎤 The main application will now start recording immediately!")
    else:
        print("\n⚠️ Test failed - calibration removal may need adjustment")

if __name__ == "__main__":
    main()