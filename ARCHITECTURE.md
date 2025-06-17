# 🎯 Avatar Generation System - Architecture Overview

This project combines SadTalker and Coqui TTS into an integrated avatar generation pipeline. Due to conflicting Python dependencies, we isolate services to maintain stability and avoid breakages.

---

## 🧠 Design Rationale

### 🧨 Problem: Dependency Conflicts
- SadTalker uses `albumentations`, which requires `numpy >= 1.24.4`
- Coqui TTS (via `k-diffusion`) requires `numpy==1.22.0`

Installing both in the same environment led to:


---

## 🛠️ Solution: Service Isolation

We split the stack into two Conda environments:

### 1. `avatar_pipeline` (SadTalker)
- Main UI
- Gradio interface
- Image + audio to video avatar generation

### 2. `tts_env` (Coqui TTS)
- Lightweight Flask API running TTS
- Input: text
- Output: WAV audio
- Exposes `/synthesize` endpoint

---

## 🔁 How They Communicate



Audio is passed back into SadTalker for final video generation.

---

## ✅ Benefits

- No dependency conflicts
- Easy debugging and modularity
- Future-proof: swap in Whisper, Bark, or other TTS models later

---

## 🚀 Future Ideas

- Add WebSocket support for faster live updates
- Move TTS to a Docker container for portability
- Cache repeated text → speech conversions

