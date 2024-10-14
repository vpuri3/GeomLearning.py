import torch
from torch import nn
from torch.nn import functional as F

# local
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, embed_dim, n_token):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, embed_dim)
        # learnable position embedding
        self.pos_embed = nn.Parameter(torch.zeros(n_token, embed_dim))

    def forward(self, tokens):
        # [B, N] -> [B, N, D]
        embed  = self.token_embedding(tokens)
        embed += self.pos_embed
        return embed
#

class CLIPLayer(nn.Module):
    def __init__(self, n_head, embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.atten = SelfAttention(n_head, embed_dim)
        self.lin1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.lin2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x): # [B, N, D]
        resid = x
        x = self.ln1(x)
        x = self.atten(x, causal_mask=True)
        x += resid

        resid = x
        x = self.ln2(x)
        x = self.lin1(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGeLU activation
        x = self.lin2(x)
        x += resid

        return x
#

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([
            ClipLayer(12, 768) for _ in range(12)
        ])
        self.ln = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        # [B, N] -> [B, N, D]
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return x
#
