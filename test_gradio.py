import gradio as gr
from PIL import Image, ImageOps

# Simple image editing function (inverts the image)
def test_edit(image, prompt):
    print("Prompt:", prompt)
    # Just to simulate processing based on prompt
    return ImageOps.invert(image.convert("RGB"))

# Build the Gradio UI
demo = gr.Interface(
    fn=test_edit,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Edit Prompt")
    ],
    outputs=gr.Image(type="pil", label="Edited Image"),
    title="🧪 Test Gradio App - Image Inverter",
    description="Upload an image and enter a prompt (not used yet) to see it inverted."
)

# Launch the app
demo.launch()
