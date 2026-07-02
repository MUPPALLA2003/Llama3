import torch

class KVCache:

    def __init__(self,num_layers:int,max_batch_size:int,max_seq_len:int,num_kv_heads:int,head_dim:int,dtype=torch.float16):

        shape = (num_layers,max_batch_size,num_kv_heads,max_seq_len,head_dim)
        self.key_cache = torch.zeros(shape,dtype=dtype)
        self.value_cache = torch.zeros(shape,dtype=dtype)
        self.seq_len = [0] * num_layers  

    def update(self,layer_idx:int,keys:torch.Tensor,values:torch.Tensor):

        batch_size, _, new_tokens, _ = keys.shape
        start = self.seq_len[layer_idx]
        end = start + new_tokens
        self.key_cache[layer_idx,:batch_size,:,start:end,:] = keys
        self.value_cache[layer_idx,:batch_size,:,start:end,:] = values
        self.seq_len[layer_idx] = end
        cached_keys = self.key_cache[layer_idx,:batch_size,:,:end,:]
        cached_values = self.value_cache[layer_idx,:batch_size,:,:end,:]

        return cached_keys, cached_values, start