from TTS.api import TTS

# Load a pre-trained English model
model_name = "tts_models/en/ljspeech/tacotron2-DDC"

# Initialize TTS
tts = TTS(model_name)

# Longer text to synthesize
text = (
    "In the rapidly evolving field of artificial intelligence, "
    "text-to-speech systems have become an essential component in "
    "making machines understand and interact with humans naturally. "
)

# Save the audio to a file
tts.tts_to_file(text=text, file_path="output.wav")
print("Audio saved as output.wav")
