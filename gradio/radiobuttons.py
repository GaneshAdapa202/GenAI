import gradio as gr

# opt_courses = [".net", "python", "java", "cloud"]

def course(courses):
    return f"The course you have selected is: {courses}"

interface = gr.Interface(
    fn = course,
    inputs = gr.Radio(
        choices = [".net", "python", "java", "cloud"], label = "select a course"
    ),
    outputs = "text",
    title = "Corses Available",
    description = "Select a course that you prefer!",

)
interface.launch()