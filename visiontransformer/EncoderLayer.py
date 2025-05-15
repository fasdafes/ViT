from torch import nn
from MultiHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden=None, drop_out=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, drop_out)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)

        hidden_dim = ffn_hidden if ffn_hidden is not None else embed_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x1 = self.layernorm1(x)
        attn_output = self.self_attn(x1)
        x2 = x + self.dropout1(attn_output)

        x3 = self.layernorm2(x2)
        ffn_output = x2 + self.dropout2(self.ffn(x3))

        return ffn_output