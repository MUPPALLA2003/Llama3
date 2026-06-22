import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
 
    def __init__(self,d_model:int,multiple_of:int = 256) -> None:

        super().__init__()

        hidden_dim = self._compute_hidden_dim(d_model,multiple_of)
        self.gate_proj = nn.Linear(d_model,hidden_dim,bias=False)
        self.up_proj = nn.Linear(d_model,hidden_dim,bias=False)
        self.down_proj = nn.Linear(hidden_dim,d_model,bias=False)

    @staticmethod
    def _compute_hidden_dim(d_model:int,multiple_of:int) -> int:
    
        hidden_dim = 4 * d_model
        hidden_dim = (2 * hidden_dim) // 3
        hidden_dim = ((hidden_dim + multiple_of - 1)// multiple_of) * multiple_of

        return hidden_dim

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)

        return output
