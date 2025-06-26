import time
import torch
from model import Transformer
from shared import context_size, device
from train_data import get_batch, estimate_loss

learning_rate = 1e-3
batch_size = 8
max_steps = 5000
eval_iters = 500

model = Transformer()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start = time.time()
print("device", device)
print("batch_size", batch_size)

for steps in range(1, max_steps + 1):
  xb, yb = get_batch("train", batch_size)

  if steps % eval_iters == 0:
    losses = estimate_loss(model, batch_size, eval_iters)
    print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  logits, cost = model(xb, yb)

  optimizer.zero_grad(set_to_none=True)
  cost.backward()
  optimizer.step()

end = time.time()
print("Minutes training:", (end - start) / 60)

torch.save(model.state_dict(), "model.pth")