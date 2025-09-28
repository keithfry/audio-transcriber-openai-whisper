import torch
import torchaudio
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import List, Union, Optional
import warnings


class WhisperTranscriber:
    def __init__(self, model_id: str = "openai/whisper-large-v3", device: Optional[str] = None):
        self.model_id = model_id

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.torch_dtype = torch.float16
            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.torch_dtype = torch.float16
            else:
                self.device = "cpu"
                self.torch_dtype = torch.float32
        else:
            self.device = device
            self.torch_dtype = torch.float16 if "cuda" in device or "mps" in device else torch.float32

        print(f"Loading Whisper model '{model_id}' on device: {self.device}")

        # Load model and processor
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)

            # Create pipeline with better configuration
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=224,  # Adjusted for token length limits
                chunk_length_s=30,
                batch_size=8,  # Reduced for stability
                return_timestamps=False,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def preprocess_audio(self, audio_data: np.ndarray, target_sample_rate: int = 16000) -> np.ndarray:
        if len(audio_data) == 0:
            return audio_data

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Remove DC offset first
        audio_data = audio_data - np.mean(audio_data)

        # More aggressive normalization to ensure adequate signal level
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            # Always normalize to use full dynamic range
            audio_data = audio_data / max_val * 0.9  # Scale to 90% to avoid clipping
            print(f"📊 Audio normalized: max was {max_val:.4f}, now 0.9000")
        else:
            print("⚠️ Audio contains only silence")
            return audio_data

        # Trim excessive silence from beginning and end
        # Find first and last non-silent samples (above 1% of max)
        threshold = 0.01
        non_silent = np.where(np.abs(audio_data) > threshold)[0]

        if len(non_silent) > 0:
            start_idx = max(0, non_silent[0] - int(0.1 * target_sample_rate))  # Keep 0.1s before
            end_idx = min(len(audio_data), non_silent[-1] + int(0.1 * target_sample_rate))  # Keep 0.1s after
            audio_data = audio_data[start_idx:end_idx]
            print(f"🔧 Trimmed audio: {len(audio_data)/target_sample_rate:.2f}s after silence removal")

        # Ensure minimum length (but don't pad excessively)
        min_length = int(0.1 * target_sample_rate)  # Minimum 0.1s
        if len(audio_data) < min_length:
            audio_data = np.pad(audio_data, (0, min_length - len(audio_data)), mode='constant')

        return audio_data

    def transcribe_audio(self, audio_data: np.ndarray, return_timestamps: bool = False) -> dict:
        if len(audio_data) == 0:
            return {"text": "", "chunks": []}

        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data)

            # Check if audio has sufficient energy and characteristics
            audio_energy = np.sqrt(np.mean(processed_audio**2))
            audio_max = np.max(np.abs(processed_audio))
            audio_std = np.std(processed_audio)

            print(f"🔊 Audio stats:")
            print(f"   Length: {len(processed_audio)/16000:.2f}s")
            print(f"   Energy (RMS): {audio_energy:.4f}")
            print(f"   Max amplitude: {audio_max:.4f}")
            print(f"   Std deviation: {audio_std:.4f}")

            if audio_energy < 0.001:
                print("⚠️ Audio energy is very low - may not contain speech")
            elif audio_energy > 0.5:
                print("⚠️ Audio energy is very high - may be clipped or noisy")

            if audio_std < 0.001:
                print("⚠️ Audio has very low variation - may be mostly silence")

            # Try direct model inference first, then fallback to pipeline
            try:
                # Direct model approach for better control
                input_features = self.processor.feature_extractor(
                    processed_audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(self.device, dtype=self.torch_dtype)

                # Force English transcription with explicit decoder start tokens
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language="en",
                    task="transcribe"
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    generated_ids = self.model.generate(
                        input_features,
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=150,
                        do_sample=False,
                        num_beams=5,
                        early_stopping=True,
                        repetition_penalty=1.1,
                        length_penalty=1.0,
                        suppress_tokens=[],  # Don't suppress any tokens
                    )

                # Decode the result
                transcription = self.processor.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                # Remove the forced tokens from the beginning if present
                transcription = transcription.strip()
                if transcription.startswith("<|startoftranscript|>"):
                    transcription = transcription.replace("<|startoftranscript|>", "").strip()
                if transcription.startswith("<|en|>"):
                    transcription = transcription.replace("<|en|>", "").strip()
                if transcription.startswith("<|transcribe|>"):
                    transcription = transcription.replace("<|transcribe|>", "").strip()

                result = {"text": transcription}
                print(f"✅ Direct model inference successful")

            except Exception as direct_error:
                print(f"⚠️ Direct model inference failed: {direct_error}")
                print("🔄 Falling back to pipeline method...")

                # Fallback to pipeline method
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self.pipe(
                        processed_audio,
                        return_timestamps=return_timestamps,
                        generate_kwargs={
                            "language": "en",
                            "task": "transcribe",
                            "do_sample": False,
                            "temperature": 0.0,
                            "num_beams": 5,
                            "repetition_penalty": 1.1,
                            "length_penalty": 1.0,
                        }
                    )

            # Ensure result has expected format
            if isinstance(result, str):
                result = {"text": result}
            elif not isinstance(result, dict):
                result = {"text": str(result)}

            return result

        except Exception as e:
            print(f"Error during transcription: {e}")
            return {"text": f"[Transcription Error: {str(e)}]", "chunks": []}

    def transcribe_chunks(self, audio_chunks: List[np.ndarray], return_timestamps: bool = False) -> List[dict]:
        results = []

        print(f"Transcribing {len(audio_chunks)} audio chunks...")

        for i, chunk in enumerate(audio_chunks):
            print(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")

            result = self.transcribe_audio(chunk, return_timestamps)
            results.append(result)

            # Print partial result
            if result["text"].strip():
                print(f"Chunk {i+1}: {result['text']}")
            else:
                print(f"Chunk {i+1}: [No speech detected]")

        return results

    def combine_transcriptions(self, transcription_results: List[dict]) -> str:
        combined_text = []

        for result in transcription_results:
            text = result.get("text", "").strip()
            if text:
                combined_text.append(text)

        return " ".join(combined_text)

    def transcribe_long_audio(self, audio_data: np.ndarray, chunk_duration: float = 30.0,
                            overlap_duration: float = 2.0, sample_rate: int = 16000) -> dict:
        if len(audio_data) == 0:
            return {"text": "", "chunks": [], "full_result": []}

        # Calculate chunk parameters
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)

        # If audio is shorter than chunk duration, transcribe directly
        if len(audio_data) <= chunk_samples:
            return self.transcribe_audio(audio_data)

        # Split into overlapping chunks
        chunks = []
        start = 0

        while start < len(audio_data):
            end = min(start + chunk_samples, len(audio_data))
            chunk = audio_data[start:end]
            chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap_samples
            if start >= len(audio_data) - overlap_samples:
                break

        print(f"Split {len(audio_data)/sample_rate:.2f}s audio into {len(chunks)} chunks")

        # Transcribe all chunks
        chunk_results = self.transcribe_chunks(chunks)

        # Combine results
        combined_text = self.combine_transcriptions(chunk_results)

        return {
            "text": combined_text,
            "chunks": chunk_results,
            "num_chunks": len(chunks),
            "total_duration": len(audio_data) / sample_rate
        }

    def get_model_info(self) -> dict:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "sample_rate": 16000,
            "max_chunk_length": 30
        }