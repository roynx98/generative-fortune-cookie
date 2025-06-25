import torch
import torch.nn as nn
from torch.nn import functional as f

class Model(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)

    if targets == None:
      return logits, None

    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    target = targets.view(B * T)
    cost = f.cross_entropy(logits, target)

    return logits, cost

  def generate(self, idx, new_tokens):
    for _ in range(new_tokens):
      logits, _ = self(idx)
      logits = logits[:, -1, :]
      probs = f.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=-1)
    return idx
