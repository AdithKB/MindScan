"""
MindScan — Flask Backend
NCI H9DAI Research Project 2026

Two modes — auto-detected at startup:

  PROXY mode  (default, no local models needed)
    Forwards /predict to the HuggingFace Space.
    Set HF_SPACE_URL env var to override the target.
    Run: python app.py

  LOCAL mode  (models/ directory present)
    Loads all 12 models from disk and runs inference locally.
    Activated automatically when models/ exists.
    Run: python app.py

Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import os, time, requests as _requests

app = Flask(__name__)

HF_SPACE_URL   = os.environ.get('HF_SPACE_URL', 'https://esvanth-mindscan.hf.space')
_LOCAL_MODELS  = os.path.join(os.path.dirname(__file__), 'models', 'classical')
_use_local     = os.path.isdir(_LOCAL_MODELS)

# ─────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  MindScan — Starting up")
print("="*55)

if _use_local:
    print("  LOCAL mode — loading models from disk...")
    from predict import load_all_models, predict_all, models_loaded
    start = time.time()
    load_all_models()
    print(f"  ✅ Models loaded in {time.time()-start:.1f}s")
else:
    print("  PROXY mode — no local models found")
    print(f"  → Forwarding requests to: {HF_SPACE_URL}")
    print("  (Download models/ from Google Drive to switch to LOCAL mode)")

print(f"  🌐 Open: http://localhost:{os.environ.get('PORT', 5001)}")
print("="*55 + "\n")


# ─────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400
    if len(text) > 5000:
        return jsonify({'error': 'Text too long (max 5000 characters)'}), 400

    if _use_local:
        # ── Local inference ───────────────────────────────────────
        if not models_loaded():
            return jsonify({'error': 'Models not ready yet — try again in a moment'}), 503
        try:
            t0 = time.time()
            result = predict_all(text)
            result['processing_time_ms'] = round((time.time() - t0) * 1000)
            return jsonify(result)
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    else:
        # ── Proxy to HuggingFace Space ────────────────────────────
        try:
            r = _requests.post(
                f'{HF_SPACE_URL}/predict',
                json={'text': text},
                timeout=120,
            )
            return r.content, r.status_code, {'Content-Type': 'application/json'}
        except _requests.exceptions.Timeout:
            return jsonify({'error': 'HuggingFace Space timed out — it may be waking up, try again in 30s'}), 504
        except _requests.exceptions.ConnectionError:
            return jsonify({'error': f'Cannot reach {HF_SPACE_URL} — check your internet connection'}), 503


@app.route('/health')
def health():
    if _use_local:
        from predict import models_loaded
        return jsonify({'status': 'ok', 'mode': 'local', 'models_ready': models_loaded()})
    else:
        try:
            r = _requests.get(f'{HF_SPACE_URL}/health', timeout=10)
            data = r.json()
            data['mode'] = 'proxy'
            data['hf_space'] = HF_SPACE_URL
            return jsonify(data)
        except Exception as e:
            return jsonify({'status': 'error', 'mode': 'proxy', 'message': str(e)}), 503


# ─────────────────────────────────────────────────────────────────
# START
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import threading, webbrowser
    port = int(os.environ.get('PORT', 5001))
    threading.Timer(1.2, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(debug=False, host='0.0.0.0', port=port)
