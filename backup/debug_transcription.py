#!/usr/bin/env python3

import numpy as np
import soundfile as sf
from audio_capture import AudioCapture
from transcriber import WhisperTranscriber
import time

def debug_transcription_pipeline():
    print("🔍 Debugging Transcription Pipeline")
    print("=" * 45)

    try:
        # Step 1: Record real audio
        print("Step 1: Recording real audio from microphone")
        audio_capture = AudioCapture()

        print("🎤 Please speak clearly for 3-5 seconds...")
        audio_data = audio_capture.record_with_vad(max_duration=15.0, min_duration=2.0)

        if len(audio_data) == 0:
            print("❌ No audio captured")
            return False

        # Save the raw audio for inspection
        raw_file = "debug_raw_audio.wav"
        sf.write(raw_file, audio_data, 16000)
        print(f"💾 Saved raw audio to {raw_file}")

        # Step 2: Analyze audio properties
        print(f"\nStep 2: Audio Analysis")
        duration = len(audio_data) / 16000
        energy = np.sqrt(np.mean(audio_data**2))
        max_amp = np.max(np.abs(audio_data))
        min_amp = np.min(audio_data)
        std_dev = np.std(audio_data)

        print(f"   Duration: {duration:.2f}s")
        print(f"   Samples: {len(audio_data)}")
        print(f"   Sample rate: 16000 Hz")
        print(f"   Data type: {audio_data.dtype}")
        print(f"   Energy (RMS): {energy:.6f}")
        print(f"   Max amplitude: {max_amp:.6f}")
        print(f"   Min amplitude: {min_amp:.6f}")
        print(f"   Std deviation: {std_dev:.6f}")
        print(f"   Dynamic range: {max_amp - min_amp:.6f}")

        # Check for common issues
        if energy < 0.001:
            print("⚠️ WARNING: Very low energy - audio may be too quiet")
        if max_amp > 0.99:
            print("⚠️ WARNING: Possible clipping detected")
        if std_dev < 0.001:
            print("⚠️ WARNING: Very low variation - may be mostly silence")

        # Step 3: Test transcription with tiny model first
        print(f"\nStep 3: Testing with Whisper Tiny (faster for debugging)")
        try:
            tiny_transcriber = WhisperTranscriber(model_id="openai/whisper-tiny")
            print("✅ Tiny model loaded successfully")

            result = tiny_transcriber.transcribe_audio(audio_data)
            print(f"📝 Tiny model result: '{result.get('text', 'No text')}'")

            if not result.get('text', '').strip():
                print("⚠️ Tiny model returned empty text")

        except Exception as e:
            print(f"❌ Tiny model failed: {e}")

        # Step 4: Test with preprocessed audio
        print(f"\nStep 4: Testing preprocessing effects")
        try:
            transcriber = WhisperTranscriber(model_id="openai/whisper-tiny")

            # Test original audio
            print("   Testing original audio...")
            original_result = transcriber.transcribe_audio(audio_data, return_timestamps=False)

            # Test preprocessed audio
            print("   Testing preprocessed audio...")
            processed_audio = transcriber.preprocess_audio(audio_data)
            processed_file = "debug_processed_audio.wav"
            sf.write(processed_file, processed_audio, 16000)
            print(f"   💾 Saved processed audio to {processed_file}")

            # Show preprocessing changes
            proc_energy = np.sqrt(np.mean(processed_audio**2))
            proc_max = np.max(np.abs(processed_audio))
            print(f"   Original energy: {energy:.6f} → Processed: {proc_energy:.6f}")
            print(f"   Original max: {max_amp:.6f} → Processed: {proc_max:.6f}")

            processed_result = transcriber.transcribe_audio(processed_audio, return_timestamps=False)

            print(f"   Original result: '{original_result.get('text', 'No text')}'")
            print(f"   Processed result: '{processed_result.get('text', 'No text')}'")

        except Exception as e:
            print(f"❌ Preprocessing test failed: {e}")

        # Step 5: Test with different audio lengths
        print(f"\nStep 5: Testing different audio lengths")
        try:
            # Test short clip (first 1 second)
            if len(audio_data) > 16000:
                short_audio = audio_data[:16000]
                short_result = transcriber.transcribe_audio(short_audio)
                print(f"   Short audio (1s): '{short_result.get('text', 'No text')}'")

            # Test middle section
            if len(audio_data) > 32000:
                mid_start = len(audio_data) // 4
                mid_end = mid_start + 16000
                mid_audio = audio_data[mid_start:mid_end]
                mid_result = transcriber.transcribe_audio(mid_audio)
                print(f"   Middle audio (1s): '{mid_result.get('text', 'No text')}'")

        except Exception as e:
            print(f"❌ Length test failed: {e}")

        # Step 6: Raw pipeline test
        print(f"\nStep 6: Raw pipeline test")
        try:
            # Test the pipeline directly without our wrapper
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Ensure audio is in correct format
                test_audio = audio_data.astype(np.float32)
                if np.max(np.abs(test_audio)) > 0:
                    test_audio = test_audio / np.max(np.abs(test_audio)) * 0.9

                print(f"   Testing raw pipeline with audio shape: {test_audio.shape}")
                print(f"   Audio range: [{np.min(test_audio):.4f}, {np.max(test_audio):.4f}]")

                raw_result = transcriber.pipe(test_audio)
                print(f"   Raw pipeline result: {raw_result}")

        except Exception as e:
            print(f"❌ Raw pipeline test failed: {e}")
            import traceback
            traceback.print_exc()

        return True

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Transcription Debug Tool")
    print("=" * 50)

    if debug_transcription_pipeline():
        print("\n🎉 Debug completed!")
        print("\n📁 Generated files:")
        print("   • debug_raw_audio.wav - Raw captured audio")
        print("   • debug_processed_audio.wav - After preprocessing")
        print("\n💡 Listen to these files to verify audio quality")
        print("💡 Check the transcription results above for clues")
    else:
        print("\n⚠️ Debug failed - check error messages above")

if __name__ == "__main__":
    main()