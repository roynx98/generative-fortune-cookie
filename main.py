import gradio as gr
from src import inference

def generate_text():
    return inference.generate(100)

demo = gr.Interface(
    fn=generate_text,
    inputs=None,
    outputs="text",
    live=False,
    title="Fortune Cookie Generator",
    description="Click the button to generate a fortune cookie message.",
    allow_flagging="never"
)

demo.launch()
