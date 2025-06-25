from src import model
import torch

with open("input/fortune-messages.txt") as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_token = { c:i for i, c in enumerate(chars) }
token_to_char = { i:c for i, c in enumerate(chars) }
encode = lambda s: [ char_to_token[c] for c in s ]
decode = lambda l: "".join([ token_to_char[t] for t in l])

model = model.Model(vocab_size=vocab_size)

def generate(new_tokens=100):
  model.eval()
  tokens = model.generate(torch.zeros(1, 1, dtype=torch.long), new_tokens)[0].tolist()
  return decode(tokens)
