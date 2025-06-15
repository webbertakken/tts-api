# TTS API

A RESTful API wrapper for [coqui-ai/TTS](https://github.com/coqui-ai/TTS) that allows text-to-speech conversion using different voices.

## Prerequisites

- Anaconda or Python 3.11
- Visual Studio 2022 (for Windows users)
- CUDA Toolkit 12.9.1 ([download](https://developer.nvidia.com/cuda-downloads))
- CuDNN 9.10.2 ([download](https://developer.nvidia.com/rdp/cudnn-archive))
- espeak-ng ([download](https://github.com/espeak-ng/espeak-ng))

## Setup

1. Create and activate virtual environment

    ```bash
    conda create --name tts python=3.11
    conda activate tts
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Install package:

    ```bash
    pip install -e .
    ```

## Running the API

Start the server with hot reloading:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## API Usage

Send a POST request to `/synthesize` with the following JSON body:

```json
{
    "text": "Hello, this is a test of the Coqui TTS API",
    "language": "en",
    "speaker": "Damien Black"
}
```

Optional parameters:
- `speaker_wav_path`: Path to a WAV file for voice cloning

## Alternative CLI Usage

You can also use the TTS CLI directly:
```bash
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --use_cuda --speaker_idx "Damien Black" --language_idx "en" --text "test 123"
```

## License

MIT License
