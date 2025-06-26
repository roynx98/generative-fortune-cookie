import gradio as gr
import inference
from model import Transformer
import torch

model = Transformer()
model.load_state_dict(torch.load("model.pth"))

def generate_text():
    return inference.generate_text(model, 100)

demo = gr.Interface(
    fn=generate_text,
    inputs=None,
    outputs="text",
    live=False,
    title="Fortune Cookie Generator",
    description="Click the button to generate a fortune cookie message.",
    flagging_mode="never",
)

demo.launch()
