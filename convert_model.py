"""
convert_model.py — Convert speech_emotion_model.h5 → TF.js format

Run this ONCE before starting the Flask server:
    pip install tensorflowjs keras
    python convert_model.py

What it does:
    1. Loads speech_emotion_model.h5 (Keras 3 / Keras 2 format)
    2. Converts to TF.js LayersModel format in ./speech_model_tfjs/
    3. Flask then serves this at /speech_model/model.json
    4. The browser loads it with tf.loadLayersModel('/speech_model/model.json')
"""
import os
import sys

MODEL_IN  = os.path.join(os.path.dirname(__file__), "speech_emotion_model.h5")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "speech_model_tfjs")

def convert():
    if not os.path.isfile(MODEL_IN):
        print(f"❌  Model not found: {MODEL_IN}")
        print("    Copy speech_emotion_model.h5 into the same folder as this script.")
        sys.exit(1)

    print(f"📦  Input  : {MODEL_IN}")
    print(f"📂  Output : {MODEL_OUT}")
    print()

    # ── Try tensorflowjs_converter (preferred) ────────────────────
    try:
        import tensorflowjs as tfjs
        import keras
        print("Loading Keras model…")
        model = keras.models.load_model(MODEL_IN)
        print(f"✅  Loaded: {model.input_shape}  →  {model.output_shape}")
        print("Converting to TF.js…")
        tfjs.converters.save_keras_model(model, MODEL_OUT)
        print(f"\n✅  Done! Model saved to: {MODEL_OUT}/")
        print("   Files created:")
        for f in os.listdir(MODEL_OUT):
            size = os.path.getsize(os.path.join(MODEL_OUT, f))
            print(f"   · {f}  ({size//1024} KB)")
        print("\n▶  Now run: python app.py")
        return
    except ImportError:
        print("⚠️  tensorflowjs not installed.")
        print("   Installing…  (pip install tensorflowjs keras)\n")

    # ── Fallback: shell out to tensorflowjs_converter CLI ─────────
    import subprocess
    result = subprocess.run(
        ["tensorflowjs_converter",
         "--input_format=keras",
         MODEL_IN, MODEL_OUT],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✅  Conversion complete via CLI.")
        print(f"   Output: {MODEL_OUT}/")
    else:
        print("❌  CLI conversion failed:")
        print(result.stderr)
        print("\nManual fix:")
        print("  pip install tensorflowjs keras")
        print("  tensorflowjs_converter --input_format=keras \\")
        print(f"      {MODEL_IN} \\")
        print(f"      {MODEL_OUT}/")
        sys.exit(1)

if __name__ == "__main__":
    convert()
