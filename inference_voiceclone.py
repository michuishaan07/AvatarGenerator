from glob import glob
import shutil
import torch
from time import strftime
import os, sys, time
import requests
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

# def synthesize_tts_audio(text, source_lang, target_lang, reference_audio=None):
#     """Send text to TTS API and return generated audio file path."""
#     tts_audio_path = os.path.join("results/audio", "tts_output.wav")
#     try:
#         data = {
#             "text": text,
#             "source_lang": source_lang,
#             "target_lang": target_lang
#         }
#         files = {"reference_audio": open(reference_audio, "rb")} if reference_audio else None
#         response = requests.post("http://localhost:8000/translate-and-speak", data=data, files=files)

#         if response.status_code == 200:
#             os.makedirs(os.path.dirname(tts_audio_path), exist_ok=True)
#             with open(tts_audio_path, "wb") as f:
#                 f.write(response.content)
#             print(f"✅ Audio synthesized and saved to {tts_audio_path}")
#             return tts_audio_path
#         else:
#             print("❌ TTS API Error:", response.text)
#     except Exception as e:
#         print("❌ TTS API request failed:", e)
#     return None

import os
import requests

def synthesize_tts_audio(text, source_lang, target_lang, reference_audio=None):
    """Send text to TTS API and return generated audio file path."""
    tts_audio_path = os.path.join("results/audio", "tts_output.wav")
    
    try:
        # Prepare payload
        data = {
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang
        }

        # Prepare file upload only if valid
        files = None
        if reference_audio and os.path.isfile(reference_audio):
            files = {"reference_audio": open(reference_audio, "rb")}
        
        # Send request to FastAPI TTS endpoint
        response = requests.post("http://localhost:8000/translate-and-speak", data=data, files=files)

        # Check for success
        if response.status_code == 200:
            os.makedirs(os.path.dirname(tts_audio_path), exist_ok=True)
            with open(tts_audio_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Audio synthesized and saved to {tts_audio_path}")
            return tts_audio_path
        else:
            print("❌ TTS API Error:", response.status_code, response.text)

    except Exception as e:
        print("❌ TTS API request failed:", str(e))

    finally:
        if files:
            files["reference_audio"].close()

    return None


def main(args):
    if not args.driven_audio and args.tts_text:
        print("🗣️ TTS text detected. Generating audio...")
        audio_path = synthesize_tts_audio(args.tts_text, args.source_lang, args.target_lang, args.reference_audio)
        if not audio_path:
            print("❌ Failed to generate audio. Exiting.")
            return
        args.driven_audio = audio_path

    if not args.driven_audio:
        print("❌ No driven_audio or tts_text provided.")
        return

    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size)

    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for eye blink reference')
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for pose reference')
            ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path = None

    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                               expression_scale=args.expression_scale, still_mode=args.still,
                               preprocess=args.preprocess, size=args.size)

    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info,
                                         enhancer=args.enhancer, background_enhancer=args.background_enhancer,
                                         preprocess=args.preprocess, img_size=args.size)

    shutil.move(result, save_dir + '.mp4')
    print('🎥 The generated video is saved as:', save_dir + '.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--driven_audio", default=None)
    parser.add_argument("--tts_text", default=None)
    parser.add_argument("--source_lang", default="en", help="Source language code (e.g., en, fr)")
    parser.add_argument("--target_lang", default="en", help="Target language code (e.g., en, fr)")
    parser.add_argument("--reference_audio", default=None, help="Reference .wav audio for voice cloning")

    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png')
    parser.add_argument("--ref_eyeblink", default=None)
    parser.add_argument("--ref_pose", default=None)
    parser.add_argument("--checkpoint_dir", default='./checkpoints')
    parser.add_argument("--result_dir", default='./results')
    parser.add_argument("--pose_style", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--expression_scale", type=float, default=1.)
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None)
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None)
    parser.add_argument('--input_roll', nargs='+', type=int, default=None)
    parser.add_argument('--enhancer', type=str, default=None)
    parser.add_argument('--background_enhancer', type=str, default=None)
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument("--face3dvis", action="store_true")
    parser.add_argument("--still", action="store_true")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--old_version", action="store_true")

    parser.add_argument('--net_recon', type=str, default='resnet50')
    parser.add_argument('--init_path', type=str, default=None)
    parser.add_argument('--use_last_fc', default=False)
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat')
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    main(args)


