#!/usr/bin/env python3

import numpy as np
from audio_capture import AudioCapture
from transcriber import WhisperTranscriber

def test_improved_transcription():
    print("🧪 Testing Improved Transcription")
    print("=" * 40)

    try:
        # Test with the large model (default in main app)
        print("Loading Whisper large-v3 model...")
        transcriber = WhisperTranscriber(model_id="openai/whisper-large-v3")

        # Test with real audio
        print("\n🎤 Recording test audio...")
        print("Please speak clearly: 'This is a test of the improved transcription system'")

        audio_capture = AudioCapture()
        audio_data = audio_capture.record_with_vad(max_duration=15.0, min_duration=2.0)

        if len(audio_data) == 0:
            print("❌ No audio captured")
            return False

        print(f"\n🤖 Transcribing with improved method...")
        result = transcriber.transcribe_audio(audio_data)

        print(f"\n📝 Transcription Result:")
        print(f"   Text: '{result.get('text', 'No text returned')}'")

        # Test a few more times with different phrases
        test_phrases = [
            "Hello, my name is Claude",
            "The weather is nice today",
            "Testing one two three four five"
        ]

        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n--- Test {i} ---")
            print(f"Please say: '{phrase}'")
            input("Press Enter when ready to record...")

            audio_data = audio_capture.record_with_vad(max_duration=10.0, min_duration=1.0)

            if len(audio_data) > 0:
                result = transcriber.transcribe_audio(audio_data)
                transcribed = result.get('text', '').strip()
                print(f"Expected: '{phrase}'")
                print(f"Got:      '{transcribed}'")

                # Basic similarity check
                expected_words = phrase.lower().split()
                transcribed_words = transcribed.lower().split()
                common_words = set(expected_words) & set(transcribed_words)

                if len(common_words) > 0:
                    accuracy = len(common_words) / len(expected_words) * 100
                    print(f"Word accuracy: {accuracy:.1f}% ({len(common_words)}/{len(expected_words)} words)")
                else:
                    print("⚠️ No matching words found")
            else:
                print("❌ No audio captured for this test")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Improved Transcription Test")
    print("=" * 50)

    print("💡 This test uses the improved transcription method:")
    print("   • Uses model.generate() instead of pipeline")
    print("   • Better generation parameters (greedy decoding)")
    print("   • Larger model for better accuracy")
    print("   • Improved audio preprocessing")

    if test_improved_transcription():
        print("\n🎉 Transcription test completed!")
        print("\n💡 If transcription is still inaccurate:")
        print("   • Ensure you're speaking clearly and loudly")
        print("   • Check that your microphone is working properly")
        print("   • Try speaking closer to the microphone")
        print("   • Minimize background noise")
    else:
        print("\n⚠️ Transcription test failed")

if __name__ == "__main__":
    main()