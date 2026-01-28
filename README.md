# Autism App

**Overview**

Autism App is a lightweight FastAPI-based screening tool that uses an ONNX acoustic model to analyze short audio recordings and produce a risk screening label (Low, High, or Uncertain) based on acoustic patterns. This repository contains the FastAPI backend, preprocessing utilities, a bundled ONNX model, and a small frontend for uploading audio files.

**Warning / Disclaimer**: This project is a research/screening aid and not a medical diagnostic. Results should not be used as a substitute for professional clinical evaluation.

**Quick Links**
- **Code:** [app/main.py](app/main.py)
- **Preprocessing:** [app/preprocessing.py](app/preprocessing.py)
- **Model:** [app/model.onnx](app/model.onnx)
- **Frontend:** [app/frontend/index.html](app/frontend/index.html)
- **Requirements:** [requirements.txt](requirements.txt)
- **Deployment manifest:** [render.yaml](render.yaml)

**Features**
- Lightweight FastAPI server with static frontend
- WebRTC VAD-based voice activity detection and MFCC extraction
- ONNX inference for fast cross-platform scoring
- Temperature-scaled softmax and uncertainty thresholding

**Repository Structure**

- app/: Backend, model and frontend static files
  - app/main.py - FastAPI app and endpoints
  - app/preprocessing.py - audio loading, VAD, MFCC extraction
  - app/model.onnx - trained ONNX model used for inference
  - app/frontend/ - static UI (index.html, script.js, style.css)
- requirements.txt - Python dependencies
- render.yaml - (optional) render.com manifest for deployment

**Requirements**

- Python 3.8+ (3.10 recommended)
- A working virtual environment (venv/virtualenv/conda)
- System audio libraries for `librosa`/`soundfile` may require OS-level packages. See notes below.

Install dependencies:

```bash
python -m venv .venv
.venv/Scripts/activate    # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

On some systems you may need to install libsndfile or other system packages for `soundfile` and `librosa`. On Windows, these are usually pre-bundled with wheels; on Linux use your distro packager (e.g., `apt install libsndfile1`).

**Run (local development)**

Start the FastAPI app using `uvicorn`:

```bash
.venv/Scripts/activate   # Windows
pip install uvicorn[standard]
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the frontend at: http://127.0.0.1:8000/ (the root route serves the static `index.html`).

**API Usage**

POST /screen — upload an audio file (form field `file`) to get a screening result.

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/screen" -F "file=@path/to/recording.wav"
```

Sample response:

```json
{
  "risk_level": "Low",
  "confidence": 0.8123,
  "probabilities": {"Low": 0.8123, "High": 0.1877},
  "disclaimer": "This is a screening tool and not a medical diagnosis..."
}
```

**How it works (brief)**

- Audio is read and normalized by app/preprocessing.py.
- WebRTC VAD filters voiced frames and sliding windows create 2-second segments.
- Each segment is converted to MFCC features shaped (1, 40, 87, 1) for model input.
- All segments are run through the ONNX model and outputs are temperature-scaled softmaxed and averaged.
- A confidence threshold is applied to mark predictions as "Uncertain" when confidence is low.

**Deployment**

This repository includes a render.yaml manifest for deploying to Render. Adjust the service settings and environment variables on your provider as needed. The app is a standard ASGI FastAPI service and can be hosted on platforms that support Python ASGI apps (Render, Heroku, Azure Web Apps, etc.).

**Troubleshooting**
- If `onnxruntime` fails to load GPU providers, fall back to CPUExecutionProvider (the app already constructs the session with CPU provider by default).
- If `librosa` or `soundfile` raise import/load errors, ensure system audio dependencies (e.g., libsndfile) are installed.
- If uploaded audio returns `No audio segments extracted`, try re-recording as a mono WAV at common sampling rates (16k/22.05k) or provide a longer clip.

**Testing**

You can test the endpoint using the frontend at the root URL or via `curl` / Postman multipart upload to `/screen`.

**Contributing**

Contributions are welcome. Please open issues for bugs or feature requests and submit pull requests for improvements. When contributing, include tests where appropriate and keep changes focused.

**License & Acknowledgements**

This project is provided as-is for research and educational purposes. If you want a specific license applied, add a LICENSE file.


