import torch
import torch.nn as nn
from RMSNorm import RMSNorm

class ResidualConnections(nn.Module):

    def __init__(self,d_model:int,dropout:float=0.0)->None:

        super().__init__()

        self.d_model = d_model
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x:torch.Tensor,sublayer)->torch.Tensor:

        return x + self.dropout(sublayer(self.norm(x)))    