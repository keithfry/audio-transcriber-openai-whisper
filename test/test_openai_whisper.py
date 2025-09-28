#!/usr/bin/env python3

import numpy as np
from audio_capture import AudioCapture
from transcriber_openai import OpenAIWhisperTranscriber

def test_openai_whisper():
    print("🧪 Testing OpenAI Whisper Implementation")
    print("=" * 45)

    try:
        # Load OpenAI Whisper (smaller model for testing)
        print("Loading OpenAI Whisper base model...")
        transcriber = OpenAIWhisperTranscriber(model_size="base")

        print("\n🎤 Recording test audio...")
        print("Please speak clearly: 'This is a test using OpenAI Whisper'")

        audio_capture = AudioCapture()
        audio_data = audio_capture.record_with_vad(max_duration=10.0, min_duration=2.0)

        if len(audio_data) == 0:
            print("❌ No audio captured")
            return False

        print(f"\n🔊 Audio captured: {len(audio_data)} samples, {len(audio_data)/16000:.2f}s")

        print(f"\n🤖 Transcribing with OpenAI Whisper...")
        result = transcriber.transcribe_audio(audio_data)

        print(f"\n📝 OpenAI Whisper Result:")
        transcribed_text = result.get('text', '').strip()
        print(f"   Text: '{transcribed_text}'")

        # Show segments if available
        segments = result.get('segments', [])
        if segments:
            print(f"   Segments: {len(segments)}")
            for i, segment in enumerate(segments[:3]):  # Show first 3
                print(f"     {i+1}: {segment.get('text', '').strip()}")

        # Analyze the result
        if transcribed_text and transcribed_text not in ['', '!', '.!', 'you', 'you!']:
            word_count = len(transcribed_text.split())
            print(f"📊 Word count: {word_count}")

            if word_count > 2:
                print("🎉 OpenAI Whisper successful! Multi-word transcription!")
                return True
            else:
                print("⚠️ Short transcription from OpenAI Whisper")
        else:
            print("❌ OpenAI Whisper also returning generic/empty response")

        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 OpenAI Whisper Test")
    print("=" * 50)

    print("💡 Testing the original OpenAI Whisper implementation:")
    print("   • Direct OpenAI Whisper library (not HuggingFace)")
    print("   • Different model loading approach")
    print("   • Alternative transcription pipeline")

    if test_openai_whisper():
        print("\n🎉 OpenAI Whisper test successful!")
        print("✅ This implementation works better than HuggingFace Transformers")

        print("\n💡 Consider switching to OpenAI Whisper for better results")
        print("   The main application can be updated to use this implementation")
    else:
        print("\n⚠️ OpenAI Whisper also had issues")
        print("💡 This suggests the problem may be:")
        print("   • Audio quality/clarity issues")
        print("   • Microphone hardware problems")
        print("   • Environmental factors (noise, distance)")
        print("   • Need for different recording settings")

if __name__ == "__main__":
    main()