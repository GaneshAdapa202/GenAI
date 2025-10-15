import gradio as gr

opt_frameworks = ["Langchain", "Langgraph", "CrewAI"]

def framework(frameworks):
    return f"you selected: {', '.join(frameworks)}"

interface = gr.Interface(
    fn = framework,
    inputs = gr.CheckboxGroup(
        choices = opt_frameworks, label = "Select Frameworks"
    ),
    outputs = "text",
    title = "checkbox",
    description = "select preffered frameworks.",
)
interface.launch()