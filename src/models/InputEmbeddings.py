import torch
import torch.nn as nn

class Embeddings(nn.Module):

    def __init__(self,vocab_size:int,d_model:int)->None:

        super().__init__()

        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,input_ids:torch.Tensor)->torch.Tensor:

        x = self.embedding(input_ids)

        return x