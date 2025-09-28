#!/usr/bin/env python3

import numpy as np
from audio_capture import AudioCapture

def test_vad_sensitivity():
    print("🧪 Testing VAD Sensitivity and Tuning")
    print("=" * 45)

    try:
        audio_capture = AudioCapture()

        print("Current VAD settings:")
        print(f"  VAD Aggressiveness: 1 (less aggressive)")
        print(f"  Silence Threshold: {audio_capture.silence_threshold}s")
        print(f"  Energy Threshold: {audio_capture.energy_threshold}")
        print(f"  Min Speech Duration: {audio_capture.min_speech_duration}s")

        print("\n🎤 Testing speech detection with current settings...")
        print("This test will record a short sample and show detection results")

        # Test recording with verbose feedback
        print("\nStarting test recording (speak for 2-3 seconds, then stay quiet)...")

        # Override some settings for testing
        audio_capture.set_silence_threshold(1.0)  # Shorter threshold for testing
        audio_capture.set_energy_threshold(0.005)  # Lower energy threshold

        audio_data = audio_capture.record_with_vad(max_duration=15.0, min_duration=1.0)

        if len(audio_data) > 0:
            duration = len(audio_data) / 16000
            energy = np.sqrt(np.mean(audio_data**2))
            print(f"\n✅ Recording completed:")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Energy: {energy:.4f}")

            # Test both detection methods on the recorded audio
            print(f"\n🔍 Testing detection methods on recorded audio:")

            # Test VAD detection
            vad_result = audio_capture.detect_speech_in_chunk(audio_data)
            print(f"   VAD Detection: {'Speech detected' if vad_result else 'No speech detected'}")

            # Test energy detection
            energy_result = audio_capture.detect_speech_by_energy(audio_data)
            print(f"   Energy Detection: {'Speech detected' if energy_result else 'No speech detected'}")

        else:
            print("❌ No audio recorded")

        return True

    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🚀 VAD Sensitivity Test")
    print("=" * 50)

    print("💡 This test helps tune the voice activity detection.")
    print("💡 If recording doesn't stop when you finish speaking:")
    print("   • The silence threshold may be too long")
    print("   • The energy threshold may be too sensitive")
    print("   • Background noise may be triggering detection")

    if test_vad_sensitivity():
        print("\n🎉 VAD sensitivity test completed!")
        print("\n🔧 If the detection isn't working well, you can adjust:")
        print("   • Silence threshold (how long to wait after speech)")
        print("   • Energy threshold (how loud audio needs to be)")
        print("   • VAD aggressiveness (0=liberal, 3=strict)")
        print("\n🎤 Try the main application now!")
    else:
        print("\n⚠️ VAD test failed.")

if __name__ == "__main__":
    main()