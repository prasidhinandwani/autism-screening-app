from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import onnxruntime as ort
import os

from app.preprocessing import preprocess_audio

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
FRONTEND = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

# Prediction labels
LABELS = ["Low", "High"]
CONFIDENCE_THRESHOLD = 0.6  # Below this, prediction is "Uncertain"

# ---------------- Temperature-scaled softmax ----------------
def softmax_with_temperature(logits, T=2.0):
    exp_logits = np.exp(logits / T)
    return exp_logits / np.sum(exp_logits)

@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND, "index.html"))

@app.post("/screen")
async def screen_audio(file: UploadFile = File(...)):
    try:
        # Read audio bytes
        audio = await file.read()

        # Preprocess to MFCC segments
        features_list = preprocess_audio(audio)  # returns list of (1, 40, 87, 1)

        if not features_list:
            return JSONResponse(
                status_code=400,
                content={"error": "No audio segments extracted from the file."}
            )

        # Concatenate segments into batch for ONNX (shape: num_segments, 40, 87, 1)
        features_batch = np.concatenate(features_list, axis=0)

        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: features_batch})[0]  # shape: (num_segments, 2)

        # ---------------- Apply temperature scaling ----------------
        outputs_scaled = np.array([softmax_with_temperature(logit, T=2.0) for logit in outputs])

        # Average softmax probabilities across segments
        avg_output = np.mean(outputs_scaled, axis=0)

        # Get predicted class and confidence
        idx = int(np.argmax(avg_output))
        confidence = float(avg_output[idx])

        # Apply threshold for uncertainty
        if confidence < CONFIDENCE_THRESHOLD:
            risk_level = "Uncertain"
        else:
            risk_level = LABELS[idx]

        return {
            "risk_level": risk_level,
            "confidence": round(confidence, 4),
            "probabilities": {LABELS[0]: round(float(avg_output[0]), 4),
                              LABELS[1]: round(float(avg_output[1]), 4)},
            "disclaimer": (
                "This is a screening tool and not a medical diagnosis. "
                "Results are based on acoustic patterns and intended to support, "
                "not replace, professional clinical evaluation."
            )
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
