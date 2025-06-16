# Usage:
#   uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Alternative (one-shot command):
#   conda activate tts
#   tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --use_cuda true --speaker_idx "Damien Black" --language_idx "en" --out_path output.wav --text "This is a test."

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from TTS.api import TTS
from typing import Optional
import base64
import io
import soundfile as sf
import traceback
import sys
import re
import json

print("Starting TTS server...")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speaker_wav_path: Optional[str] = None
    speaker: str = "Damien Black"

class PhonemeInfo(BaseModel):
    phoneme: str
    start_time: float
    end_time: float
    duration: float

def get_phonemes_from_text(text: str) -> list:
    # Simple phoneme approximation based on text
    # This is a basic implementation - you might want to use a more sophisticated approach
    text = text.lower()
    # Split into words and then into characters
    words = text.split()
    phonemes = []
    for word in words:
        # Add word boundary
        phonemes.append(' ')
        # Add characters as phonemes
        for char in word:
            if char.isalpha():
                phonemes.append(char)
    print(f"Generated phonemes: {phonemes}")
    return phonemes

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    try:
        print(f"\nReceived request with text: {request.text}")

        # Create a bytes buffer to store the audio
        audio_buffer = io.BytesIO()

        # Generate speech
        if request.speaker_wav_path:
            wav = tts.tts(
                text=request.text,
                speaker_wav=request.speaker_wav_path,
                language=request.language
            )
        else:
            wav = tts.tts(
                text=request.text,
                language=request.language,
                speaker=request.speaker
            )

        # Save the audio to buffer using soundfile
        sf.write(audio_buffer, wav, tts.synthesizer.output_sample_rate, format='WAV')

        # Calculate timing information
        total_duration = len(wav) / tts.synthesizer.output_sample_rate
        print(f"Total audio duration: {total_duration} seconds")

        # Get phonemes from text
        phonemes = get_phonemes_from_text(request.text)
        print(f"Number of phonemes: {len(phonemes)}")

        # Create phoneme-level timing information
        phoneme_info = []
        current_time = 0.0

        # Calculate average phoneme duration
        avg_phoneme_duration = total_duration / len(phonemes)
        print(f"Average phoneme duration: {avg_phoneme_duration} seconds")

        for phoneme in phonemes:
            # Estimate duration based on phoneme type
            if phoneme == ' ':  # Word boundary
                duration = avg_phoneme_duration * 0.5
            elif phoneme in ['a', 'e', 'i', 'o', 'u']:
                duration = avg_phoneme_duration * 1.5
            else:
                duration = avg_phoneme_duration * 0.8

            end_time = current_time + duration

            phoneme_info.append(PhonemeInfo(
                phoneme=phoneme,
                start_time=current_time,
                end_time=end_time,
                duration=duration
            ))

            current_time = end_time

        print(f"Generated phoneme info: {[p.dict() for p in phoneme_info[:5]]}")  # Print first 5 phonemes

        # Get the audio data and encode it to base64
        audio_data = audio_buffer.getvalue()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Create the response with explicit phonemes array
        response = {
            "status": "success",
            "audioBase64": audio_base64,
            "audioMime": "audio/wav",
            "totalDuration": total_duration,
            "phonemes": [p.dict() for p in phoneme_info],  # Explicitly include phonemes
            "metadata": {
                "sampleRate": tts.synthesizer.output_sample_rate,
                "language": request.language,
                "speaker": request.speaker if not request.speaker_wav_path else "custom",
                "numPhonemes": len(phonemes),
                "avgPhonemeDuration": avg_phoneme_duration
            }
        }

        # Print response structure for debugging
        print("\nResponse structure:")
        print(f"Status: {response['status']}")
        print(f"Total duration: {response['totalDuration']}")
        print(f"Number of phonemes in response: {len(response['phonemes'])}")
        print(f"First few phonemes: {response['phonemes'][:5]}")
        print(f"Metadata: {json.dumps(response['metadata'], indent=2)}")

        return response
    except Exception as e:
        # Get detailed error information
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"Error details: {error_info}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        )
