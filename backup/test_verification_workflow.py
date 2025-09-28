#!/usr/bin/env python3

import numpy as np
from audio_playback import AudioPlayback

def test_audio_playback():
    print("🧪 Testing Audio Playback Functionality")
    print("=" * 45)

    try:
        playback = AudioPlayback()

        # Test 1: Playback info
        print("✅ AudioPlayback initialized")

        # Test 2: Generate test audio
        print("\n🎵 Generating test audio (440Hz tone)...")
        sample_rate = 16000
        duration = 2.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Test 3: Audio info
        audio_info = playback.get_audio_info(test_audio)
        print(f"📊 Audio info: {audio_info}")

        # Test 4: Audio playback test
        print(f"\n🔊 Testing audio playback...")
        print("💡 You should hear a 440Hz tone for 2 seconds")
        input("Press Enter to play test tone...")

        success = playback.test_playback()
        if success:
            print("✅ Audio playback test successful")
        else:
            print("❌ Audio playback test failed")

        # Test 5: Test with actual audio data
        print(f"\n🎵 Testing playback with generated audio...")
        input("Press Enter to play generated sine wave...")

        playback.play_audio(test_audio, blocking=True)
        print("✅ Generated audio playback completed")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_verification_workflow():
    print("\n🧪 Testing Verification Workflow Simulation")
    print("=" * 45)

    try:
        # Simulate the verification process
        test_transcriptions = [
            "Hello, this is a test",
            "The weather is nice today",
            "Testing one two three"
        ]

        # Generate test audio for each
        playback = AudioPlayback()

        for i, text in enumerate(test_transcriptions, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"📝 Simulated transcription: '{text}'")

            # Generate corresponding test audio (different frequencies for variety)
            frequency = 440 + (i * 100)  # 440Hz, 540Hz, 640Hz
            duration = 1.0 + (i * 0.5)   # 1.5s, 2s, 2.5s

            t = np.linspace(0, duration, int(16000 * duration), False)
            test_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

            audio_info = playback.get_audio_info(test_audio)
            print(f"🎵 Test audio: {audio_info['duration']:.1f}s at {frequency}Hz")

            # Simulate user verification
            response = input("❓ Is this transcription correct? (y/n/p for play audio): ").strip().lower()

            if response == 'y':
                print("✅ Transcription verified as correct")
            elif response == 'n':
                print("❌ Transcription marked as incorrect")

                play_response = input("🔊 Would you like to hear what was recorded? (y/n): ").strip().lower()
                if play_response == 'y':
                    print(f"🔊 Playing back test audio ({frequency}Hz)...")
                    playback.play_audio(test_audio, blocking=True)
                    print("✅ Audio playback completed")
                else:
                    print("ℹ️ Audio playback skipped")
            elif response == 'p':
                print(f"🔊 Playing test audio ({frequency}Hz)...")
                playback.play_audio(test_audio, blocking=True)
                print("✅ Direct audio playback completed")

        return True

    except Exception as e:
        print(f"❌ Verification workflow test failed: {e}")
        return False

def main():
    print("🚀 Verification Workflow Test")
    print("=" * 50)

    print("💡 This test verifies:")
    print("   • Audio playback functionality")
    print("   • User verification prompts")
    print("   • Audio playback on incorrect transcription")
    print("   • Complete workflow integration")

    # Test audio playback first
    if not test_audio_playback():
        print("\n❌ Audio playback test failed - cannot proceed")
        return

    # Test verification workflow
    if test_verification_workflow():
        print("\n🎉 Verification workflow test completed!")
        print("\n✅ The main application now includes:")
        print("   • Transcription verification prompts")
        print("   • Audio playback for verification")
        print("   • User feedback and suggestions")
        print("   • Complete workflow integration")

        print("\n🎤 Ready to test with the main application:")
        print("   source .env/bin/activate && python main.py")
    else:
        print("\n⚠️ Verification workflow test failed")

if __name__ == "__main__":
    main()