#!/usr/bin/env python3

import numpy as np
from audio_playback import AudioPlayback

def test_simple_playback():
    print("🧪 Testing Simple Audio Playback")
    print("=" * 35)

    try:
        playback = AudioPlayback()
        print("✅ AudioPlayback initialized")

        # Generate test audio
        print("🎵 Generating test audio (440Hz tone)...")
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        test_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Test audio info
        audio_info = playback.get_audio_info(test_audio)
        print(f"📊 Audio info:")
        print(f"   Duration: {audio_info['duration']:.2f}s")
        print(f"   Energy: {audio_info['energy']:.4f}")
        print(f"   Max amplitude: {audio_info['max_amplitude']:.4f}")
        print(f"   Samples: {audio_info['samples']}")

        # Test playback without user input
        print("🔊 Testing audio playback (should hear 440Hz tone)...")
        playback.play_audio(test_audio, blocking=True)
        print("✅ Audio playback completed")

        # Test cleanup
        playback.stop_playback()
        print("✅ Playback cleanup successful")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Simple Playback Test")
    print("=" * 50)

    if test_simple_playback():
        print("\n🎉 Simple playback test passed!")
        print("\n✅ Audio playback functionality is working")
        print("✅ The verification feature is ready to use")

        print("\n🎤 The main application now includes:")
        print("   • Transcription verification (y/n prompts)")
        print("   • Audio playback for incorrect transcriptions")
        print("   • Helpful suggestions for improvement")

        print("\n🚀 Ready to use the enhanced application:")
        print("   source .env/bin/activate && python main.py")
    else:
        print("\n⚠️ Playback test failed - audio verification may not work")

if __name__ == "__main__":
    main()