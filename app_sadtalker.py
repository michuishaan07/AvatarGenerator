import os
import sys
import gradio as gr
from src.gradio_demo import SadTalker

try:
    import webui
    in_webui = True
except ImportError:
    in_webui = False

def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("## 🗣️ SadTalker - Talking Avatar Demo")

        with gr.Row():
            with gr.Column():
                source_image = gr.Image(label="Source Image", type="filepath")
                driven_audio = gr.Audio(label="Input Audio", type="filepath")
                input_text = gr.Textbox(
                    label="Or generate from text (TTS)", 
                    lines=3,
                    placeholder="Type text here..."
                )
                tts_button = gr.Button("🎧 Generate Audio from Text")
                tts_audio_path = gr.Textbox(visible=False)

                def call_tts_api(text):
                    import requests
                    response = requests.post("http://localhost:5002/synthesize", json={"text": text})
                    output_path = "/tmp/generated_tts.wav"
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return output_path, output_path

                tts_button.click(
                    fn=call_tts_api,
                    inputs=[input_text],
                    outputs=[driven_audio, tts_audio_path]
                )

            with gr.Column():
                gr.Markdown("🔧 Configure generation options below.")
                pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose Style", value=0)
                size_of_image = gr.Radio([256, 512], value=256, label="Image Resolution")
                preprocess_type = gr.Radio(['crop', 'resize', 'full', 'extcrop', 'extfull'], value='crop', label="Preprocess Type")
                is_still_mode = gr.Checkbox(label="Still Mode (Less head motion)")
                batch_size = gr.Slider(label="Batch size", step=1, maximum=10, value=2)
                enhancer = gr.Checkbox(label="Use GFPGAN Face Enhancer")
                submit = gr.Button("Generate Talking Video", variant="primary")
                gen_video = gr.Video(label="Generated Video", format="mp4")

        fn_to_call = warpfn(sad_talker.test) if warpfn else sad_talker.test

        submit.click(
            fn=fn_to_call,
            inputs=[
                source_image,
                driven_audio,
                preprocess_type,
                is_still_mode,
                enhancer,
                batch_size,
                size_of_image,
                pose_style
            ],
            outputs=[gen_video]
        )

        return sadtalker_interface

if __name__ == "__main__":
    demo = sadtalker_demo()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860,share=True)
