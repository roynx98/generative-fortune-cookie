import torch
from shared import decode

@torch.no_grad()
def generate_text(model, new_tokens):
  model.eval()
  tokens = model.generate(torch.zeros(1, 1, dtype=torch.long), new_tokens)[0].tolist()
  return decode(tokens)
