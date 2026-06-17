import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self,d_model:int,eps:float=1e-05)->None:

        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model,dtype = torch.float32))

    def forward(self,x:torch.Tensor)->torch.Tensor:

        t,dtype = x.float(),x.dtype
        t = (t * torch.rsqrt(torch.mean(t**2,dim=-1,keepdim=True)+self.eps)).type_as(x)

        return t*self.scale