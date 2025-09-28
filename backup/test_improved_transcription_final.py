#!/usr/bin/env python3

import numpy as np
from audio_capture import AudioCapture
from transcriber import WhisperTranscriber

def test_improved_transcription():
    print("🧪 Testing Improved Transcription Pipeline")
    print("=" * 45)

    try:
        # Use the large model for better accuracy
        print("Loading Whisper large-v3 model...")
        transcriber = WhisperTranscriber(model_id="openai/whisper-large-v3")

        print("\n🎤 Recording test audio...")
        print("Please speak clearly: 'Hello, this is a test of the speech recognition system'")

        audio_capture = AudioCapture()
        audio_data = audio_capture.record_with_vad(max_duration=15.0, min_duration=2.0)

        if len(audio_data) == 0:
            print("❌ No audio captured")
            return False

        print(f"\n🔊 Audio captured: {len(audio_data)} samples, {len(audio_data)/16000:.2f}s")

        # Show original audio stats
        energy = np.sqrt(np.mean(audio_data**2))
        max_amp = np.max(np.abs(audio_data))
        print(f"📊 Original audio - Energy: {energy:.6f}, Max: {max_amp:.6f}")

        print(f"\n🤖 Transcribing with improved pipeline...")
        result = transcriber.transcribe_audio(audio_data)

        print(f"\n📝 Transcription Result:")
        transcribed_text = result.get('text', '').strip()
        print(f"   Text: '{transcribed_text}'")

        # Analyze the result
        if transcribed_text and transcribed_text not in ['', '!', '.!', 'you', 'you!']:
            print("✅ Transcription appears to contain real content!")

            word_count = len(transcribed_text.split())
            print(f"📊 Word count: {word_count}")

            if word_count > 2:
                print("🎉 Multi-word transcription successful!")
                return True
            else:
                print("⚠️ Short transcription - may still be hallucinating")
        else:
            print("❌ Transcription appears to be hallucinated/generic")
            print("💡 This suggests the audio is not reaching the model properly")

        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Improved Transcription Test")
    print("=" * 50)

    print("💡 Testing improvements:")
    print("   • Better audio normalization (full dynamic range)")
    print("   • Silence trimming to focus on speech")
    print("   • Direct model inference with forced tokens")
    print("   • Beam search with repetition penalty")
    print("   • Large-v3 model for better accuracy")

    if test_improved_transcription():
        print("\n🎉 Improved transcription test passed!")
        print("✅ The transcription pipeline should now work much better")

        print("\n🎤 Try the main application:")
        print("   source .env/bin/activate && python main.py")
    else:
        print("\n⚠️ Transcription still not working optimally")
        print("💡 If issues persist, this may be due to:")
        print("   • Microphone input level too low")
        print("   • Background noise interfering")
        print("   • Whisper model hallucination on unclear audio")
        print("   • Need for different microphone or audio setup")

if __name__ == "__main__":
    main()