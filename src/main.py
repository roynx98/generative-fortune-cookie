import gradio as gr
import inference
from model import Transformer
import torch

model = Transformer()
model.load_state_dict(torch.load("model.pth"))

def generate_text():
    return inference.generate_text(model, 500)

custom_css = """
.gradio-container {
    align-self: center;
    width: 100%;
    max-width: 600px;
}
.center-text {
    text-align: center;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
        # Generative Fortune Cookie
    """, elem_classes="center-text")

    gr.Markdown("""
        This app generates fortune cookie messages using a transformer model built from scratch with PyTorch, check out the code on [GitHub](https://github.com/roynx98/generative-fortune-cookie).
    """)

    with gr.Column():
        text1 = gr.Textbox(interactive=False, show_label=False, placeholder="Your fortune cookie message will appear here...")

        clear_btn = gr.Button("Clear")
        clear_btn.click(fn=lambda: "", outputs=text1)

        generate_btn = gr.Button("Generate", variant="primary")
        generate_btn.click(fn=lambda: generate_text(), outputs=text1)
    gr.Markdown(
        "Made by [Roy Rodriguez](https://royrodriguez.me/)",
        elem_classes="center-text"
    )

demo.launch()
