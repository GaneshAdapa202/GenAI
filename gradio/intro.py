import gradio as gr

def greet(name):
    return f"Hello, {name}!"

interface = gr.Interface(
    fn = greet,
    inputs = "text",
    outputs = "text",
    title = "GREET",
    description = "Greetings"
)

# interface = gr.Interface(
#     fn = greet,
#     inputs = gr.Textbox(lines = 1, placeholder = "Enter name"),
#     outputs = "text",
#     title = "Wellcome",
#     description = "Ganesh"
# )

# interface = gr.Interface(
#     fn = greet,
#     inputs = gr.Textbox(lines=1, placeholder="Enter your name here "),
#     outputs = gr.Textbox(lines=1, placeholder="Your output"),
#     title = "Greeting",
#     description = "Greeting app"
# )
interface.launch()