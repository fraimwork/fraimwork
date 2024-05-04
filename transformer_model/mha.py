import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads

        assert (self.d_k * self.num_heads) == self.d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def compute_scaled_dot_product_attention(self, query, key, value, key_padding_mask = None, attention_mask = None):
        min_val = torch.finfo(query.dtype).min
        B, _, _, d_k = query.shape

        x = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).eq(1)
            x = x.masked_fill(mask, min_val)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2).eq(1)
            x = x.masked_fill(mask, min_val)
        
        x = torch.matmul(torch.softmax(x, dim=-1), value).transpose(2,1)

        x = x.contiguous().view(B, -1, self.num_heads * self.d_k)

        return x
    
    def compute_mh_qkv_transformation(self, Q, K, V):
        B, T_q, _ = Q.shape
        _, T_k, _ = K.shape
        _, T_v, _ = V.shape
        
        q = self.W_q(Q).contiguous().view(B, T_q ,self.num_heads, self.d_k).transpose(1,2)
        k = self.W_k(K).contiguous().view(B, T_k, self.num_heads, self.d_k).transpose(1,2)
        v = self.W_v(V).contiguous().view(B, T_v, self.num_heads, self.d_k).transpose(1,2)

        return q, k, v
    
    def forward(self, query, key, value, key_padding_mask = None, attention_mask = None):
        q, k, v = self.compute_mh_qkv_transformation(query, key, value)
        return self.W_o(self.compute_scaled_dot_product_attention(q, k, v, key_padding_mask=key_padding_mask, attention_mask=attention_mask))
