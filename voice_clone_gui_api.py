from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil, os, uuid, subprocess

app = FastAPI()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index_voice_clone.html", {"request": request})


@app.post("/generate")
async def generate_talking_avatar(
    request: Request,
    source_image: UploadFile = File(None),
    driven_audio: UploadFile = File(None),
    ref_eyeblink: UploadFile = File(None),
    ref_pose: UploadFile = File(None),
    reference_audio: UploadFile = File(None),

    tts_text: str = Form(None),
    source_lang: str = Form("en"),
    target_lang: str = Form("en"),

    pose_style: str = Form(None),
    enhancer: str = Form(None),
    background_enhancer: str = Form(None),
    still: bool = Form(False)
):
    # Create temp directory
    temp_dir = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)

    def save_upload(upload, name):
        if upload and upload.filename:
            file_path = os.path.join(temp_dir, upload.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
            return file_path.replace("\\", "/")
        return None


    # Save files
    img_path = save_upload(source_image, "source_image")
    audio_path = save_upload(driven_audio, "driven_audio")
    eyeblink_path = save_upload(ref_eyeblink, "ref_eyeblink")
    pose_path = save_upload(ref_pose, "ref_pose")
    ref_audio_path = save_upload(reference_audio, "reference_audio")

    # Construct arguments for CLI
    args = ["python", "inference_voiceclone.py"]

    if img_path:
        args.extend(["--source_image", img_path])
    if audio_path:
        args.extend(["--driven_audio", audio_path])
    if eyeblink_path:
        args.extend(["--ref_eyeblink", eyeblink_path])
    if pose_path:
        args.extend(["--ref_pose", pose_path])
    if ref_audio_path:
        args.extend(["--reference_audio", ref_audio_path])
    if tts_text:
        args.extend(["--tts_text", tts_text])
        args.extend(["--source_lang", source_lang])
        args.extend(["--target_lang", target_lang])
    if pose_style:
        args.extend(["--pose_style", pose_style])
    if enhancer:
        args.extend(["--enhancer", enhancer])
    if background_enhancer:
        args.extend(["--background_enhancer", background_enhancer])
    if still:
        args.append("--still")

    args.extend(["--result_dir", temp_dir])

    # Run subprocess
    subprocess.run(args)

    # Get output video
    video_path = None
    for f in os.listdir(temp_dir):
        if f.endswith(".mp4"):
            video_path = os.path.join(temp_dir, f).replace("\\", "/")
            break

    if not video_path:
        return JSONResponse(status_code=400, content={"error": "No video generated."})

    # Return JSON for frontend
    relative_path = os.path.relpath(video_path, UPLOAD_FOLDER).replace("\\", "/")
    return {
        "video_url": f"/uploads/{relative_path}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

