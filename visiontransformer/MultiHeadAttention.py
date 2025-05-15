import torch
from torch import nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,drop_out =0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.drop_out1 = nn.Dropout(drop_out)
        self.drop_out2 = nn.Dropout(drop_out)

        self.q_proj = nn.Linear(embed_dim,embed_dim)
        self.k_proj = nn.Linear(embed_dim,embed_dim)
        self.v_proj = nn.Linear(embed_dim,embed_dim)
        self.out_proj = nn.Linear(embed_dim,embed_dim)

    def forward(self,x):
        batch_size,seq_len,embed_dim = x.shape
        q = self.q_proj(x).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(x).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        attn_score = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        attn_prob = F.softmax(attn_score,dim=-1)
        attn_prob = self.drop_out1(attn_prob)
        attn_output = torch.matmul(attn_prob,v)
        attn_output = attn_output.transpose(1,2).reshape(batch_size,seq_len,embed_dim)
        output = self.out_proj(attn_output)
        output = self.drop_out2(output)
        return output