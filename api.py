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

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    try:
        # Create a bytes buffer to store the audio
        audio_buffer = io.BytesIO()

        if request.speaker_wav_path:
            tts.tts_to_file(
                text=request.text,
                speaker_wav=request.speaker_wav_path,
                language=request.language,
                file_path=audio_buffer
            )
        else:
            tts.tts_to_file(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
                file_path=audio_buffer
            )

        # Get the audio data and encode it to base64
        audio_data = audio_buffer.getvalue()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        return {
            "status": "success",
            "audioBase64": audio_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
