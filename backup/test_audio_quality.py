#!/usr/bin/env python3

import numpy as np
import soundfile as sf
from audio_capture import AudioCapture
from transcriber import WhisperTranscriber
import time

def test_audio_capture_quality():
    print("🧪 Testing Audio Capture Quality")
    print("=" * 40)

    try:
        # Initialize audio capture
        audio_capture = AudioCapture()
        print("✅ Audio capture initialized")

        # Record a short test
        print("\n🎤 Recording 3 seconds of audio for quality test...")
        print("Please speak clearly into the microphone!")

        audio_data = audio_capture.record_with_vad(max_duration=10.0, min_duration=1.0)

        if len(audio_data) == 0:
            print("❌ No audio captured")
            return False

        # Save raw audio to file for inspection
        output_file = "test_recording.wav"
        sf.write(output_file, audio_data, 16000)
        print(f"💾 Saved raw audio to {output_file}")

        # Analyze audio properties
        duration = len(audio_data) / 16000
        energy = np.sqrt(np.mean(audio_data**2))
        max_val = np.max(np.abs(audio_data))
        std_val = np.std(audio_data)

        print(f"\n📊 Raw Audio Analysis:")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Samples: {len(audio_data)}")
        print(f"   Energy (RMS): {energy:.4f}")
        print(f"   Max amplitude: {max_val:.4f}")
        print(f"   Std deviation: {std_val:.4f}")
        print(f"   Data type: {audio_data.dtype}")

        # Test with tiny model for faster feedback
        print(f"\n🤖 Testing transcription with tiny model...")
        transcriber = WhisperTranscriber(model_id="openai/whisper-tiny")
        result = transcriber.transcribe_audio(audio_data)

        print(f"\n📝 Transcription Result:")
        print(f"   Text: '{result.get('text', 'No text returned')}'")

        # Test preprocessing effects
        processed_audio = transcriber.preprocess_audio(audio_data)
        processed_file = "test_recording_processed.wav"
        sf.write(processed_file, processed_audio, 16000)
        print(f"💾 Saved processed audio to {processed_file}")

        print(f"\n✅ Test completed! Check the .wav files to verify audio quality.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🚀 Audio Quality Test")
    print("=" * 50)

    if test_audio_capture_quality():
        print("\n🎉 Audio quality test completed!")
        print("\n💡 Check the generated .wav files:")
        print("   • test_recording.wav - Raw captured audio")
        print("   • test_recording_processed.wav - After preprocessing")
        print("\n🔍 If transcription is still poor, the issue may be:")
        print("   • Microphone input level too low/high")
        print("   • Background noise")
        print("   • Audio device configuration")
        print("   • VAD processing artifacts")
    else:
        print("\n⚠️ Audio quality test failed.")

if __name__ == "__main__":
    main()