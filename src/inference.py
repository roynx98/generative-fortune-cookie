import torch
from shared import decode, device

@torch.no_grad()
def generate_sentence(model):
  model.eval()
  tokens = model.generate(torch.zeros(1, 1, dtype=torch.long, device=device))[0].tolist()
  tokens = tokens[1:]
  return decode(tokens)
