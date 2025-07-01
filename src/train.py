import time
import torch
from model import Transformer
from shared import device
from train_data import get_batch, estimate_loss
from inference import generate_sentence
import boto3
from config import BUCKET_NAME, OBJECT_KEY

def train():
    learning_rate = 1e-4
    batch_size = 64
    max_steps = 10000
    eval_iters = 500

    model = Transformer()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start = time.time()
    print("device", device)
    print("batch_size", batch_size)

    for steps in range(1, max_steps + 1):
        xb, yb = get_batch("train", batch_size)

        if steps % eval_iters == 0:
            losses = estimate_loss(model, batch_size, eval_iters)
            print(
                f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        logits, cost = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        cost.backward()
        optimizer.step()

    end = time.time()
    print("Minutes training:", (end - start) / 60)
    print("Sentence example:", generate_sentence(model))

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    train()
