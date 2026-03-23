# EmotionSense AI — Multimodal Emotion Detection

## What's new in this version
- **Voice analysis** now uses your trained **Speech CNN** (`speech_emotion_model.h5`)
- CNN runs **fully in the browser** via TensorFlow.js — no server round-trips for audio
- Automatic fallback to YIN heuristic if model isn't converted yet
- Face CNN unchanged (pure numpy, server-side)

## Architecture
```
Audio → MediaRecorder → Web Audio API → 32×32 Mel-Spectrogram → Speech CNN (TF.js) → 7 emotions
Image → Webcam → Canvas → JPEG → Flask /predict → Face CNN (numpy) → 7 emotions
Face + Voice → Weighted Fusion (60%/40%) → Final Verdict
```

## Speech CNN details
- Architecture: Conv2D(16) → BN → MaxPool → Conv2D(32) → BN → MaxPool → Dense(64) → Dense(7)
- Input: **32×32 mel-spectrogram** (3 seconds of audio, normalized to [0,1])
- Output: 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- Trained with Keras 3.9.2

## Setup

### Step 1 — Install dependencies
```bash
pip install flask numpy opencv-python-headless tensorflowjs keras
```

### Step 2 — Convert speech model (run once)
Place `speech_emotion_model.h5` in this folder, then:
```bash
python convert_model.py
```
This creates `speech_model_tfjs/` which Flask serves to the browser.

### Step 3 — Start the server
```bash
python app.py
# Open http://localhost:5000
```

## Usage
1. **Step 1:** Click "Capture & Analyze Face" — look at the camera
2. **Step 2:** Click "Start Recording" — speak naturally for 2–4 seconds, then stop
3. **Step 3:** Read the AI Verdict in the right panel

## Tips for best results
- Good front lighting (face the light source)
- Look directly at the camera
- Speak clearly and expressively
- Exaggerate emotions slightly for better detection

## File structure
```
emotion_final_2/
├── app.py                  ← Flask backend (face CNN + model serving)
├── convert_model.py        ← One-time model converter (H5 → TF.js)
├── detect.html             ← Main UI (speech CNN via TF.js)
├── index.html              ← Landing page
├── model_weights.json      ← Face CNN weights (JSON)
├── emotion_cnn_model.h5    ← Face CNN (Keras)
├── speech_emotion_model.h5 ← Speech CNN ← place this here
├── requirements.txt
└── speech_model_tfjs/      ← Created by convert_model.py
    ├── model.json
    └── group1-shard1of1.bin
```
