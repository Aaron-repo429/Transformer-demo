import torch
from torch import nn

import torch.nn.functional as F
import math

random_torch=torch.rand(4,4)
print(random_torch)


from torch import Tensor
#将输入的词汇表索引转换为指定维度的Embedding

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size,d_model):
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model,max_len,device):
        super(PositionalEmbedding,self).__init__()
        self.encoding=torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad =False
        pos=torch.arange(0,max_len,device=device)
        pos=pos.float().unsqueeze(dim=1)
        _2i=torch.arange(0,d_model,step=2,device=device).float()
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_/d_model)))
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_/d_model)))

    def forward(self,x):
        batch_size,seq_len=x.size()
        return self.encoding[:seq_len,:]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size,d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb =TokenEmbedding(vocab_size,d_model)
        self.pos_emb=PositionalEmbedding(d_model,max_len,device)
        self.drop_out=nn.Dropout(p=drop_prob)

    def forward(self,x):
        tok_emb=self.tok_emb(x)
        pos_emb=self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)
