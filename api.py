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

class SegmentInfo(BaseModel):
    text: str
    start_time: float
    end_time: float
    duration: float

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    try:
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
        # Split text into words for basic timing
        words = request.text.split()
        total_duration = len(wav) / tts.synthesizer.output_sample_rate
        avg_word_duration = total_duration / len(words)

        # Create word-level timing information
        segments = []
        current_time = 0.0

        for word in words:
            # Estimate duration based on word length
            # This is a simple approximation - you might want to adjust the scaling
            duration = len(word) * (avg_word_duration / 5)  # Adjust the divisor to fine-tune timing
            end_time = current_time + duration

            segments.append(SegmentInfo(
                text=word,
                start_time=current_time,
                end_time=end_time,
                duration=duration
            ))

            current_time = end_time

        # Get the audio data and encode it to base64
        audio_data = audio_buffer.getvalue()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        return {
            "status": "success",
            "audioBase64": audio_base64,
            "totalDuration": total_duration,
            "segments": [s.dict() for s in segments],
            "metadata": {
                "sampleRate": tts.synthesizer.output_sample_rate,
                "language": request.language,
                "speaker": request.speaker if not request.speaker_wav_path else "custom",
                "numWords": len(words),
                "avgWordDuration": avg_word_duration
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
