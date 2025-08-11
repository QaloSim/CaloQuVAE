"""
Transformer-based decoder module for a neural network, implementing multi-head self-attention mechanisms.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(self.fn(x))
        return x

class Head(nn.Module):
    '''
    Self-attention block
    '''
    def __init__(self, dim, head_size=16):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(dim,head_size, bias=False)
        self.query = nn.Linear(dim,head_size, bias=False)
        self.value = nn.Linear(dim,head_size, bias=False)
        # self.ln1 = nn.LayerNorm(dim)
        # self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, l, h, w = x.shape
        x = rearrange(x, "b c l h w -> b (l h w) c")
        # x = self.ln1(x)
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)
        # out = self.ln2(wei @ v)
        out = wei @ v
        
        return rearrange(out, "b (l h w) c -> b c l h w",l=l,h=h,w=w)

    
class Multihead(nn.Module):
    '''
        Multi-head attention
    '''
    def __init__(self, dim, num=1, head_size=16):
        super().__init__()
        self.heads = nn.ModuleList([Head(dim,head_size) for _ in range(num)])
        
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=1)
    
class Headv2(nn.Module):
    '''
    Self-attention block
    '''
    def __init__(self, dim, head_size=16):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(dim,head_size, bias=False)
        self.query = nn.Linear(dim,head_size, bias=False)
        self.value = nn.Linear(dim,head_size, bias=False)
        

    def forward(self, x):
        # b, c, l, h, w = x.shape
        # x = rearrange(x, "b c l h w -> b (l h w) c")
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5
        wei = F.softmax(wei,dim=-1)
        v = self.value(x)
        out = wei @ v
        
        return out
    
class Multiheadv2(nn.Module):
    '''
        Multi-head attention
    '''
    def __init__(self, dim, num=1):
        super().__init__()
        head_size = dim // num
        self.heads = nn.ModuleList([Headv2(dim,head_size) for _ in range(num)])
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        b, c, l, h, w = x.shape
        x = rearrange(x, "b c l h w -> b (l h w) c")
        x = torch.cat([h(self.ln1(x)) for h in self.heads], dim=2)
        x = self.ln2(x)
        return rearrange(x, "b (l h w) c -> b c l h w",l=l,h=h,w=w)