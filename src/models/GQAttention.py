import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def repeat_kv(x: torch.Tensor,n_rep: int) -> torch.Tensor:
    
    batch_size, n_kv_heads, seq_len, head_dim = x.shape

    x = (x[:, :, None, :, :].expand(batch_size,n_kv_heads,n_rep,seq_len,head_dim).reshape(batch_size,n_kv_heads * n_rep,seq_len,head_dim))

    return x


class GQAttention(nn.Module):
   
    def __init__(self,d_model: int,n_heads: int,max_seq_len: int,dropout: float = 0.1,n_kv_heads: Optional[int] = None,use_flash_attention: bool = True):

        super().__init__()

        if d_model % n_heads != 0:

            raise ValueError("d_model must be divisible by n_heads.")

        if n_kv_heads is None:

            n_kv_heads = n_heads

        if n_heads % n_kv_heads != 0:
            
            raise ValueError("n_heads must be divisible by n_kv_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.max_seq_len = max_seq_len

        self.use_flash_attention = (use_flash_attention and hasattr(F,"scaled_dot_product_attention"))

        self.wq = nn.Linear(d_model,n_heads * self.head_dim,bias=False)
        self.wk = nn.Linear(d_model,n_kv_heads * self.head_dim,bias=False)
        self.wv = nn.Linear(d_model,n_kv_heads * self.head_dim,bias=False)
        self.wo = nn.Linear(d_model,d_model,bias=False)
        self.attention_dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.ones(max_seq_len,max_seq_len,dtype=torch.bool),diagonal=1)
        self.register_buffer("causal_mask",causal_mask,persistent=False)