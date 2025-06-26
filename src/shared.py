import torch

with open("input/fortune-messages.txt") as f:
  text = f.read()

chars = sorted(list(set(text)))
char_to_token = { c:i for i, c in enumerate(chars) }
token_to_char = { i:c for i, c in enumerate(chars) }

encode = lambda s: [ char_to_token[c] for c in s ]
decode = lambda l: "".join([ token_to_char[t] for t in l])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(chars)
n_embeddings = 32
context_size = 8
dropout = 0.2
