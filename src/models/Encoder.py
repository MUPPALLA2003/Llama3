import torch
import torch.nn as nn
from typing import Optional
from .GQAttention import GQAttention
from .ResidualConnections import ResidualConnections
from .FeedForward import FeedForward

class LlamaEncoder(nn.Module):

    def __init__(self,d_model:int,n_heads:int,max_seq_len:int,dropout:float = 0.1,n_kv_heads:Optional[int] = None,rope_base:float = 500000.0,multiple_of:int = 256) -> None:

        super().__init__()

        self.attention = GQAttention(d_model,n_heads,max_seq_len,dropout,n_kv_heads,rope_base)
        self.mlp = FeedForward(d_model,multiple_of)
        self.residual_connection = nn.ModuleList([ResidualConnections(d_model,dropout) for i in range(2)])

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        x = self.residual_connection[0](x,self.attention)
        x = self.residual_connection[1](x,self.mlp)

        return x

