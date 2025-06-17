from flask import Flask, request, jsonify, send_file
from TTS.api import TTS
import tempfile
import os

app = Flask(__name__)

# List of reliable alternative models (ordered by recommendation)
ALTERNATIVE_MODELS = [
    "tts_models/en/ljspeech/tacotron2-DDC",  # Most stable English model
    "tts_models/en/ljspeech/glow-tts",       # Good quality, fast
    "tts_models/en/vctk/vits",               # Multi-speaker English
    "tts_models/en/jenny/jenny"              # Clean female voice
]

# Initialize TTS with the first working model
tts = None
for model in ALTERNATIVE_MODELS:
    try:
        tts = TTS(model_name=model)
        print(f"Successfully loaded model: {model}")
        break
    except Exception as e:
        print(f"Failed to load {model}: {str(e)}")

if tts is None:
    raise RuntimeError("Could not load any TTS model")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    # Validate input
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Create temp file
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        # Handle different model requirements
        if "vctk" in tts.model_name:  # Multi-speaker model
            tts.tts_to_file(
                text=text,
                speaker=tts.speakers[0],
                file_path=temp_path
            )
        else:  # Standard single-speaker model
            tts.tts_to_file(
                text=text,
                file_path=temp_path
            )

        return send_file(
            temp_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name="speech.wav"
        )
        
    except Exception as e:
        return jsonify({"error": f"TTS synthesis failed: {str(e)}"}), 500
        
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)