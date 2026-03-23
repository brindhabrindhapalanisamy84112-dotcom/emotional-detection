"""
EmotionSense AI — Flask Backend
Face CNN (numpy inference) + Speech CNN (served to browser via TF.js)
"""
from __future__ import annotations
import os, base64, json
import numpy as np
import cv2
from flask import Flask, jsonify, request, send_from_directory

APP_DIR          = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH     = os.path.join(APP_DIR, "model_weights.json")
SPEECH_MODEL_DIR = os.path.join(APP_DIR, "speech_model_tfjs")

app = Flask(__name__, static_folder=APP_DIR, static_url_path="")

# ── Face CNN weights ───────────────────────────────────────────────
print("Loading face model weights…")
_raw = json.load(open(WEIGHTS_PATH))
W = {}
for k in _raw['weights']:
    raw = base64.b64decode(_raw['weights'][k])
    W[k] = np.frombuffer(raw, dtype=np.float32).copy().reshape(_raw['shapes'][k])
print("Face model loaded. Layers:", list(W.keys()))

LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Pure numpy face inference ──────────────────────────────────────
def relu(x):    return np.maximum(0, x)
def softmax(x): e = np.exp(x - x.max()); return e / e.sum()

def conv2d_valid(x, kernel, bias):
    from numpy.lib.stride_tricks import as_strided
    H, W_, C = x.shape
    kH, kW, _, F = kernel.shape
    oH, oW = H - kH + 1, W_ - kW + 1
    s = x.strides
    patches = as_strided(x, shape=(oH, oW, kH, kW, C),
                         strides=(s[0], s[1], s[0], s[1], s[2]))
    out = patches.reshape(oH * oW, kH * kW * C) @ kernel.reshape(kH * kW * C, F) + bias
    return out.reshape(oH, oW, F)

def maxpool2(x):
    H, W_, C = x.shape
    H2, W2 = H // 2, W_ // 2
    return x[:H2*2, :W2*2, :].reshape(H2, 2, W2, 2, C).max(axis=(1, 3))

def run_face_model(face32_gray):
    x = face32_gray.astype(np.float32) / 255.0
    x = x.reshape(32, 32, 1)
    x = relu(conv2d_valid(x, W['c1k'], W['c1b']))
    x = maxpool2(x)
    x = relu(conv2d_valid(x, W['c2k'], W['c2b']))
    x = maxpool2(x)
    x = x.flatten()
    x = relu(x @ W['d1k'] + W['d1b'])
    x = x @ W['d2k'] + W['d2b']
    return softmax(x)

# ── Face detection + preprocessing ────────────────────────────────
def detect_and_preprocess(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = None
    for (sf, mn) in [(1.1, 5), (1.05, 4), (1.08, 3), (1.15, 3)]:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=sf, minNeighbors=mn, minSize=(28, 28),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            break
    if faces is None or len(faces) == 0:
        return None, "No face detected. Ensure good lighting and face the camera directly."
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.10 * max(w, h))
    x1 = max(0, x - pad);         y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad); y2 = min(gray.shape[0], y + h + pad)
    face = gray[y1:y2, x1:x2]
    face32 = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    face32 = clahe.apply(face32)
    return face32, None

# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def index():  return send_from_directory(APP_DIR, "index.html")

@app.get("/detect")
@app.get("/detect.html")
def detect(): return send_from_directory(APP_DIR, "detect.html")

@app.route("/speech_model/<path:filename>")
def serve_speech_model(filename):
    if not os.path.isdir(SPEECH_MODEL_DIR):
        return jsonify({
            "error": "Speech model not converted. Run: python convert_model.py"
        }), 404
    return send_from_directory(SPEECH_MODEL_DIR, filename)

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image file required"}), 400
    file = request.files["image"]
    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400
    face32, err = detect_and_preprocess(image)
    if err:
        return jsonify({"error": err}), 400
    probs = run_face_model(face32)
    idx = int(np.argmax(probs))
    return jsonify({
        "emotion": LABELS[idx],
        "confidence": round(float(probs[idx]) * 100, 1),
        "all_probs": {l: round(float(p) * 100, 1) for l, p in zip(LABELS, probs)}
    })

if __name__ == "__main__":
    if not os.path.isdir(SPEECH_MODEL_DIR):
        print("\n⚠️  Speech CNN not converted yet.")
        print("   Run:  python convert_model.py")
        print("   Then restart the server.\n")
    else:
        print(f"✅  Speech CNN model found at: {SPEECH_MODEL_DIR}")
    app.run(debug=True, port=5000)
