import gradio as gr
import inference
from model import Transformer
import torch
from config import MODEL_LOCAL_FILE_PATH
from shared import device

def main():
    model = Transformer()
    model.load_state_dict(torch.load(MODEL_LOCAL_FILE_PATH, map_location=torch.device(device)))

    custom_css = """
    .gradio-container {
        align-self: center;
        width: 100%;
        max-width: 800px;
    }
    .center-text {
        text-align: center;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown(
            """
            # Generative Fortune Cookie
        """,
            elem_classes="center-text",
        )

        gr.Markdown(
            """
            This app generates fortune cookie messages using a transformer model built from scratch with PyTorch, check out the code on [GitHub](https://github.com/roynx98/generative-fortune-cookie).
        """
        )

        with gr.Column():
            text1 = gr.Textbox(
                interactive=False,
                lines=3,
                show_label=False,
                placeholder="Your fortune cookie message will appear here...",
            )

            clear_btn = gr.Button("Clear")
            clear_btn.click(fn=lambda: "", outputs=text1)

            generate_btn = gr.Button("Generate", variant="primary")
            generate_btn.click(
                fn=lambda: inference.generate_sentence(model), outputs=text1
            )
        gr.Markdown(
            "Made by [Roy Rodriguez](https://royrodriguez.me/)",
            elem_classes="center-text",
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
