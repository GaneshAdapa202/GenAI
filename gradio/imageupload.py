import gradio as gr

def save_img(image):
    image.save("Uploaded_image.png")
    return "Image save as uploaded_image.png"

interface = gr.Interface(
    fn = save_img,
    inputs = gr.Image(type="pil", label="Upload an Image"),
    outputs = "text",
    title = "Image",
    description = "Upload N image"
)
interface.launch()