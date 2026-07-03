import torch
import torch.nn as nn
import torch.nn.functional as F
from .PositionalEmbeddings import precompute_rope_params,apply_rope
from typing import Optional
import math

def repeat_kv(x:torch.Tensor,n_rep:int) -> torch.Tensor:
  
    if n_rep == 1:

        return x
    
    batch,n_kv_heads,seq_len,head_dim = x.shape

    return (
        x[:, :, None, :, :]
        .expand(batch,n_kv_heads,n_rep,seq_len,head_dim)
        .reshape(batch,n_kv_heads * n_rep,seq_len,head_dim)
    )
class GQAttention(nn.Module):

    def __init__(self,d_model:int,n_heads:int,max_seq_len:int,dropout:float = 0.1,n_kv_heads:Optional[int] = None,rope_base:float = 500000.0):

        super().__init__()

        assert d_model % n_heads == 0

        n_kv_heads = n_kv_heads or n_heads

        assert n_heads % n_kv_heads == 0
 
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_model = d_model
        self.dropout = dropout
        self.wq = nn.Linear(d_model,n_heads * self.head_dim,bias=False)
        self.wk = nn.Linear(d_model,n_kv_heads * self.head_dim,bias=False)
        self.wv = nn.Linear(d_model,n_kv_heads * self.head_dim,bias=False)
        self.wo = nn.Linear(d_model,d_model,bias=False)

        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        cos, sin = precompute_rope_params(self.head_dim, max_seq_len, rope_base)
        self.register_buffer("cos_cache", cos, persistent=False) 
        self.register_buffer("sin_cache", sin, persistent=False)
 
    def forward(self,x: torch.Tensor,kv_cache=None,layer_idx: Optional[int] = None,attention_mask: Optional[torch.Tensor] = None):

        batch, seq_len, _ = x.shape
        q = self.wq(x).view(batch, seq_len, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is None:
       
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)     
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)

            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        else:

            assert layer_idx is not None, "layer_idx required with kv_cache"
 
            start_pos = kv_cache.seq_len[layer_idx]
            cos = self.cos_cache[start_pos : start_pos + seq_len]
            sin = self.sin_cache[start_pos : start_pos + seq_len]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)     
            k, v, _ = kv_cache.update(layer_idx, k, v)
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
            s_q  = q.shape[-2]
            s_kv = k.shape[-2]
            offset = s_kv - s_q

            mask = (
                torch.arange(s_kv, device=x.device)[None, :]
                > torch.arange(s_q, device=x.device)[:, None] + offset
            )
 
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(mask, float("-inf"))
            output = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype) @ v
 
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        return self.wo(output)