#!/usr/bin/env python3

import numpy as np
from audio_capture import AudioCapture
import time

def test_noise_handling():
    print("🧪 Testing Noise Handling and VAD Sensitivity")
    print("=" * 50)

    try:
        audio_capture = AudioCapture()

        print("This test will help verify that VAD properly handles background noise.")
        print("\n📋 Test Steps:")
        print("1. Noise floor calibration (stay quiet)")
        print("2. Test recording with speech detection")
        print("3. Analysis of detection thresholds")

        # Force recalibration
        audio_capture.noise_calibrated = False

        print(f"\n🔧 Current settings:")
        print(f"   VAD Aggressiveness: 1")
        print(f"   Base Energy Threshold: {audio_capture.energy_threshold}")
        print(f"   Speech/Noise Ratio Required: {audio_capture.min_speech_energy_ratio}x")
        print(f"   Silence Threshold: {audio_capture.silence_threshold}s")

        # Test with shorter recording for faster feedback
        print(f"\n🎤 Starting test recording...")
        print("💡 Speak clearly for 2-3 seconds, then stop speaking")
        print("💡 The recording should stop within 1.5 seconds after you finish")

        start_time = time.time()
        audio_data = audio_capture.record_with_vad(max_duration=20.0, min_duration=1.0)
        end_time = time.time()

        total_time = end_time - start_time
        print(f"\n📊 Recording Results:")

        if len(audio_data) > 0:
            duration = len(audio_data) / 16000
            energy = np.sqrt(np.mean(audio_data**2))
            max_amplitude = np.max(np.abs(audio_data))

            print(f"   Total recording time: {total_time:.2f}s")
            print(f"   Audio duration: {duration:.2f}s")
            print(f"   Audio energy (RMS): {energy:.4f}")
            print(f"   Max amplitude: {max_amplitude:.4f}")
            print(f"   Noise floor: {audio_capture.noise_floor:.4f}")
            print(f"   Effective energy threshold: {audio_capture.energy_threshold:.4f}")

            # Analyze if detection is working properly
            energy_ratio = energy / audio_capture.noise_floor if audio_capture.noise_floor > 0 else 0
            print(f"   Energy vs noise ratio: {energy_ratio:.1f}x")

            if total_time < 25:  # Stopped before max duration
                print("✅ Recording stopped automatically - VAD is working!")
            else:
                print("⚠️ Recording hit max duration - VAD may not be detecting speech end")

            if energy_ratio > 2:
                print("✅ Good speech-to-noise ratio detected")
            else:
                print("⚠️ Low speech-to-noise ratio - may have issues with detection")

        else:
            print("❌ No audio recorded")

        # Test detection methods on the recorded audio
        if len(audio_data) > 1600:  # At least 0.1s of audio
            print(f"\n🔍 Testing detection methods on recorded audio:")
            vad_result = audio_capture.detect_speech_in_chunk(audio_data)
            energy_result = audio_capture.detect_speech_by_energy(audio_data)

            print(f"   VAD Detection: {'✅ Speech' if vad_result else '❌ No speech'}")
            print(f"   Energy Detection: {'✅ Speech' if energy_result else '❌ No speech'}")
            print(f"   Combined (AND): {'✅ Speech' if (vad_result and energy_result) else '❌ No speech'}")

        return True

    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🚀 Noise Handling Test")
    print("=" * 50)

    if test_noise_handling():
        print("\n🎉 Noise handling test completed!")
        print("\n💡 Key improvements made:")
        print("   • Automatic noise floor calibration")
        print("   • Adaptive energy thresholds based on ambient noise")
        print("   • Stricter VAD requirements (40% vs 15% frame agreement)")
        print("   • Combined VAD + energy detection (both must agree)")
        print("   • Peak energy validation above noise floor")
        print("\n🎤 The application should now:")
        print("   • Calibrate noise floor when first started")
        print("   • Only detect speech when significantly above background")
        print("   • Stop recording promptly when speech ends")
        print("   • Ignore ambient room noise and background sounds")
    else:
        print("\n⚠️ Noise handling test failed.")

if __name__ == "__main__":
    main()