import torch
import torch.nn as nn
from einops import rearrange



class MultiHeadAttention(nn.Module):
    def __init__(self, dim, dim_head, heads, dropout, max_len):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not ( heads == 1 and dim_head == dim )

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm = nn.LayerNorm(dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        causal_attn_mask = torch.tril(torch.ones(max_len, max_len)).bool()
        self.register_buffer("causal_attn_mask", causal_attn_mask)

    
    def forward(self, x):
        b, n, d = x.shape

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.einsum("bhid, bhjd -> bhij", q, k) * self.scale

        mask = self.causal_attn_mask[:n, :n]
        dots = dots.masked_fill(~mask, float('-inf'))

        attn = self.softmax(dots)
        attn = self.dropout(attn)

        out = torch.einsum("bhij, bhjd -> bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        
        out = self.to_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
        

class Decoder(nn.Module):
    def __init__(self, *, dim, depth, dim_head, heads, hidden_dim, dropout, max_len):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    MultiHeadAttention(dim, dim_head, heads, dropout, max_len),
                    FeedForward(dim, hidden_dim, dropout)
                ])
            )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        for mha, ff in self.layers:
            x = mha(x) + x
            x = ff(x) + x
        return self.norm(x)
    

if __name__ == "__main__":
    dec = Decoder(dim=4, depth=1, dim_head=2, heads=2, hidden_dim=16, dropout=0., max_len=10)
    x = torch.rand(1, 5, 4)
    print(dec(x).shape)