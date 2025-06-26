import torch
from shared import device, encode, context_size, text

data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(source, batch_size):
  data = train_data if source == "train" else val_data
  rv = torch.randint(len(data) - context_size, (batch_size,))
  inputs = torch.stack([data[r:r+context_size] for r in rv])
  outputs = torch.stack([data[r+1:r+context_size+1] for r in rv])
  inputs, outputs = inputs.to(device), outputs.to(device)

  return [inputs, outputs]

@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters):
  out = {}
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
      xb, yb = get_batch(split, batch_size)
      logits, cost = model(xb, yb)
      losses[i] = cost.item()
    out[split] = losses.mean()
  model.train()
  return out