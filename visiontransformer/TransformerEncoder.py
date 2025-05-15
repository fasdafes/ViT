from torch import nn
from EncoderLayer import EncoderLayer

class TransformerEncoder(nn.Module):
  def __init__(self,embed_dim,num_heads,ffn_hidden=None,num_layers=12,drop_out=0.1):
    super().__init__()
    self.layers = nn.ModuleList([
        EncoderLayer(embed_dim,num_heads,ffn_hidden,drop_out) for _ in range(num_layers)
    ])
    self.norm = nn.LayerNorm(embed_dim)
  def forward(self,x):
    for layer in self.layers:
      x = layer(x)
    return self.norm(x)