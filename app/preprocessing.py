import numpy as np
import librosa
import soundfile as sf
import io
import webrtcvad

TARGET_SR = 22050
SEGMENT_SECONDS = 2.0
SEGMENT_SAMPLES = int(TARGET_SR * SEGMENT_SECONDS)

N_MFCC = 40
TIME_STEPS = 87
N_FFT = 1024
HOP_LENGTH = 512

# Initialize WebRTC VAD
vad = webrtcvad.Vad(3)

# ---------------- LOAD AUDIO ----------------
def load_audio(audio_bytes):
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    
    # Convert multi-channel to mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    
    # Resample to TARGET_SR if needed
    if sr != TARGET_SR:
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=TARGET_SR)
    
    # Normalize
    max_val = np.max(np.abs(wav))
    if max_val > 0:
        wav = wav / max_val
    else:
        wav = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
    
    return wav

# ---------------- VAD ----------------
def apply_vad(wav):
    # webrtcvad requires 16 kHz
    wav_16k = librosa.resample(y=wav, orig_sr=TARGET_SR, target_sr=16000)
    
    frame_len = int(16000 * 0.03)  # 30 ms frames at 16 kHz
    voiced = []
    
    for i in range(0, len(wav_16k), frame_len):
        frame = wav_16k[i:i+frame_len]
        if len(frame) == frame_len:
            pcm = (frame * 32768).astype(np.int16).tobytes()
            try:
                if vad.is_speech(pcm, 16000):
                    voiced.append(frame)
            except Exception:
                continue  # skip frame if VAD fails
    
    if voiced:
        # Concatenate and resample back to TARGET_SR
        voiced_audio = np.concatenate(voiced)
        if 16000 != TARGET_SR:
            voiced_audio = librosa.resample(y=voiced_audio, orig_sr=16000, target_sr=TARGET_SR)
        return voiced_audio
    else:
        return wav  # fallback: original audio

# ---------------- PREPROCESS ----------------
def preprocess_audio(audio_bytes):
    # Load and normalize
    wav = load_audio(audio_bytes)
    
    # Apply VAD safely
    wav = apply_vad(wav)

    # ---------------- Sliding windows ----------------
    segments = []
    step = SEGMENT_SAMPLES // 2  # 50% overlap
    for start in range(0, len(wav), step):
        seg = wav[start:start+SEGMENT_SAMPLES]
        if len(seg) < SEGMENT_SAMPLES:
            seg = np.pad(seg, (0, SEGMENT_SAMPLES - len(seg)))
        segments.append(seg)

    # ---------------- MFCC extraction ----------------
    mfcc_segments = []
    for seg in segments:
        mfcc = librosa.feature.mfcc(
            y=seg,
            sr=TARGET_SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        # Pad/truncate time steps
        if mfcc.shape[1] < TIME_STEPS:
            mfcc = np.pad(mfcc, ((0, 0), (0, TIME_STEPS - mfcc.shape[1])))
        mfcc = mfcc[:, :TIME_STEPS]

        # Expand dims: (1, 40, 87, 1)
        mfcc = mfcc[:, :, np.newaxis]
        mfcc = mfcc[np.newaxis, :, :, :].astype(np.float32)
        mfcc_segments.append(mfcc)

    # Return list of segments
    return mfcc_segments
