#!/usr/bin/env python3

import numpy as np
import soundfile as sf
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from audio_capture import AudioCapture

def debug_whisper_processing():
    print("🔍 Debugging Whisper Model Input Processing")
    print("=" * 50)

    try:
        # Step 1: Record real audio
        print("Step 1: Recording real audio...")
        audio_capture = AudioCapture()
        audio_data = audio_capture.record_with_vad(max_duration=10.0, min_duration=2.0)

        if len(audio_data) == 0:
            print("❌ No audio captured")
            return False

        # Save original for reference
        sf.write("debug_original.wav", audio_data, 16000)
        print(f"💾 Saved original audio: {len(audio_data)} samples, {len(audio_data)/16000:.2f}s")

        # Step 2: Test different preprocessing approaches
        print("\nStep 2: Testing different preprocessing approaches...")

        # Original approach
        print("\n--- Original Preprocessing ---")
        processed_original = preprocess_original(audio_data)
        sf.write("debug_processed_original.wav", processed_original, 16000)

        # Librosa approach
        print("\n--- Librosa Preprocessing ---")
        processed_librosa = preprocess_librosa(audio_data)
        sf.write("debug_processed_librosa.wav", processed_librosa, 16000)

        # Minimal preprocessing
        print("\n--- Minimal Preprocessing ---")
        processed_minimal = preprocess_minimal(audio_data)
        sf.write("debug_processed_minimal.wav", processed_minimal, 16000)

        # Step 3: Test with different models
        print("\nStep 3: Testing transcription with different approaches...")

        models_to_test = [
            "openai/whisper-tiny",
            "openai/whisper-base",
        ]

        for model_id in models_to_test:
            print(f"\n--- Testing {model_id} ---")

            try:
                # Load model components
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,  # Use float32 for better compatibility
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                processor = AutoProcessor.from_pretrained(model_id)

                # Test each preprocessing approach
                for name, audio in [
                    ("Original", processed_original),
                    ("Librosa", processed_librosa),
                    ("Minimal", processed_minimal)
                ]:
                    print(f"  Testing {name} preprocessing...")

                    # Direct feature extraction and inference
                    try:
                        inputs = processor(
                            audio,
                            sampling_rate=16000,
                            return_tensors="pt"
                        )

                        # Generate transcription
                        with torch.no_grad():
                            generated_ids = model.generate(
                                inputs.input_features,
                                max_new_tokens=100,
                                do_sample=False,
                                language="en",
                                task="transcribe"
                            )

                        # Decode result
                        transcription = processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )[0]

                        print(f"    Result: '{transcription}'")

                    except Exception as e:
                        print(f"    Error: {e}")

            except Exception as e:
                print(f"  Failed to load {model_id}: {e}")

        # Step 4: Test with a known good audio sample
        print("\nStep 4: Testing with synthesized speech-like audio...")
        test_synthetic_audio()

        return True

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_original(audio_data):
    """Original preprocessing from transcriber.py"""
    if len(audio_data) == 0:
        return audio_data

    # Convert to float32 if needed
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Remove DC offset
    audio_data = audio_data - np.mean(audio_data)

    # Conservative normalization
    max_val = np.max(np.abs(audio_data))
    if max_val > 0.95:
        audio_data = audio_data * (0.95 / max_val)

    # Ensure minimum length
    min_length = int(0.5 * 16000)
    if len(audio_data) < min_length:
        audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode='constant')

    return audio_data

def preprocess_librosa(audio_data):
    """Librosa-based preprocessing"""
    # Ensure float32
    audio_data = audio_data.astype(np.float32)

    # Use librosa for resampling and normalization
    audio_data = librosa.util.normalize(audio_data)

    # Trim silence
    audio_data, _ = librosa.effects.trim(audio_data, top_db=20)

    return audio_data

def preprocess_minimal(audio_data):
    """Minimal preprocessing - just format conversion"""
    # Only ensure correct data type and basic normalization
    audio_data = audio_data.astype(np.float32)

    # Simple normalization
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val * 0.9

    return audio_data

def test_synthetic_audio():
    """Test with synthetic speech-like audio"""
    print("🎵 Testing with synthetic speech patterns...")

    # Create speech-like audio with multiple formants
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Simulate speech formants (typical frequencies for vowels)
    f1, f2, f3 = 800, 1200, 2400  # Typical for 'a' sound

    # Create formant-based audio
    speech_like = (
        0.3 * np.sin(2 * np.pi * f1 * t) +
        0.2 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )

    # Add amplitude modulation (speech envelope)
    envelope = np.sin(2 * np.pi * 5 * t) * 0.5 + 0.5
    speech_like = speech_like * envelope

    # Convert to float32
    speech_like = speech_like.astype(np.float32)

    # Save for inspection
    sf.write("debug_synthetic_speech.wav", speech_like, 16000)
    print(f"💾 Saved synthetic speech audio")

    # Test transcription
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-tiny",
            torch_dtype=torch.float32
        )
        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")

        inputs = processor(speech_like, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                max_new_tokens=50,
                do_sample=False
            )

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"🎤 Synthetic audio transcription: '{transcription}'")

    except Exception as e:
        print(f"❌ Synthetic audio test failed: {e}")

def main():
    print("🚀 Whisper Input Debug Tool")
    print("=" * 50)

    print("💡 This will help identify why Whisper returns '!' instead of actual text")
    print("💡 Please speak clearly when prompted to record")

    if debug_whisper_processing():
        print("\n🎉 Debug completed!")
        print("\n📁 Generated debug files:")
        print("   • debug_original.wav - Original captured audio")
        print("   • debug_processed_*.wav - Different preprocessing approaches")
        print("   • debug_synthetic_speech.wav - Synthetic test audio")
        print("\n💡 Listen to these files and check the transcription results above")
    else:
        print("\n⚠️ Debug failed")

if __name__ == "__main__":
    main()