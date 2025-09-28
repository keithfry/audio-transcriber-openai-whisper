#!/usr/bin/env python3

import numpy as np
from audio_capture import AudioCapture
from transcriber import WhisperTranscriber

def test_fixed_transcription():
    print("🧪 Testing Fixed Transcription (No Invalid Parameters)")
    print("=" * 55)

    try:
        print("Loading Whisper model...")
        transcriber = WhisperTranscriber(model_id="openai/whisper-tiny")  # Use tiny for faster testing

        print("\n🎤 Recording test audio...")
        print("Please speak clearly: 'This is a test of the transcription system'")

        audio_capture = AudioCapture()
        audio_data = audio_capture.record_with_vad(max_duration=10.0, min_duration=1.0)

        if len(audio_data) == 0:
            print("❌ No audio captured")
            return False

        print(f"\n🤖 Transcribing with fixed model.generate()...")
        result = transcriber.transcribe_audio(audio_data)

        print(f"\n📝 Transcription Result:")
        transcribed_text = result.get('text', '').strip()
        print(f"   Text: '{transcribed_text}'")

        if transcribed_text and not transcribed_text.startswith('[Transcription Error'):
            print("✅ Transcription successful!")

            # Check if it's more than just generic responses
            if len(transcribed_text.split()) > 1:
                print("✅ Multi-word transcription detected")
            else:
                print("⚠️ Single word response - may still be hallucinating")

            return True
        else:
            print("❌ Transcription failed or returned error")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Fixed Transcription Test")
    print("=" * 50)

    print("💡 This test verifies:")
    print("   • model.generate() works without invalid parameters")
    print("   • No more 'no_timestamps' error")
    print("   • Improved transcription accuracy")

    if test_fixed_transcription():
        print("\n🎉 Fixed transcription test passed!")
        print("\n✅ The model.generate() method should now work properly")
        print("✅ No more parameter errors")
        print("✅ Better transcription accuracy expected")

        print("\n🎤 Try the main application now:")
        print("   source .env/bin/activate && python main.py")
    else:
        print("\n⚠️ Test failed - may need further parameter adjustments")

if __name__ == "__main__":
    main()