import torch

def precompute_rope_params(head_dim:int,max_seq_len:int,base:float = 10000.0):
    
    i = torch.arange(0, head_dim, 2).float()
    inv_freq = 1.0 / (base ** (i / head_dim))  #
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, inv_freq)  
    angles = torch.cat([angles, angles], dim=-1)  

    return angles.cos(), angles.sin() 


def rotate_half(x:torch.Tensor) -> torch.Tensor:
    
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x:torch.Tensor,cos:torch.Tensor,sin:torch.Tensor) -> torch.Tensor:
 
    return (x * cos) + (rotate_half(x) * sin)