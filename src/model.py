import torch
import torch.nn as nn
from torch.nn import functional as f

from shared import vocab_size, context_size, n_embeddings, dropout


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(context_size, n_embeddings)
        self.blocks = nn.Sequential(
            TransformerBlock(n_embeddings, 6),
            TransformerBlock(n_embeddings, 6),
            TransformerBlock(n_embeddings, 6),
            TransformerBlock(n_embeddings, 6),
            TransformerBlock(n_embeddings, 6),
            TransformerBlock(n_embeddings, 6),
        )
        self.lm_head = nn.Linear(n_embeddings, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tokens_embeddings = self.token_embedding_table(idx)
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )
        x = tokens_embeddings + pos_embeddings
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets == None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        target = targets.view(B * T)
        cost = f.cross_entropy(logits, target)
        return logits, cost

    def generate(self, idx):
        while True:
            logits, _ = self(idx[:, -context_size:])
            logits = logits[:, -1, :]
            probs = f.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            stop_token = 5
            should_stop = idx_next[0, -1] == stop_token
            if should_stop:
                return idx

            idx = torch.cat((idx, idx_next), dim=1)


class SingleHeadOfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = f.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadOfAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [SingleHeadOfAttention(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(res))


class FeedForward(nn.Module):
    def __init__(self, n_embeddings):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.net = nn.Sequential(
            nn.Linear(n_embeddings, n_embeddings * 4),
            nn.ReLU(),
            nn.Linear(n_embeddings * 4, n_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embeddings, n_head):
        super().__init__()
        head_size = n_embeddings // n_head
        self.sa = MultiHeadOfAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embeddings)
        self.n1 = nn.LayerNorm(n_embeddings)
        self.n2 = nn.LayerNorm(n_embeddings)

    def forward(self, x):
        x = x + self.sa(self.n1(x))
        x = x + self.ffwd(self.n2(x))
        return x
