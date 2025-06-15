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
        output_path = "output.wav"

        if request.speaker_wav_path:
            tts.tts_to_file(
                text=request.text,
                speaker_wav=request.speaker_wav_path,
                language=request.language,
                file_path=output_path
            )
        else:
            tts.tts_to_file(
                text=request.text,
                language=request.language,
                speaker=request.speaker,
                file_path=output_path
            )

        return {"status": "success", "file_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
