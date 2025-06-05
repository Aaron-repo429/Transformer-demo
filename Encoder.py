import torch.nn.functional as F
import math
import torch.nn as nn

from Attention import MultiHeadAttention
from Embedding import TransformerEmbedding


class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x.self.dropout(x)
        x=self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.attention=MultiHeadAttention(d_model,n_head, dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,dropout)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout2=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        _x=x
        x=self.attention(x,x,x,mask)
        x=self.dropout1(x)
        x=self.norm1(x+_x)
        _x=x
        x=self.ffn(x)
        x=self.dropout2(x)
        x=self.norm2(x+_x)
        return x

class Encoder(nn.Module):
    def __init__(self,enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,device,dropout=0.1):
        super(Encoder,self)._init_()
        # 修正TransformerEmbedding参数顺序
        self.embedding = TransformerEmbedding(
            vocab_size=enc_voc_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=dropout,
            device=device
        )
        # self.embeddingTransformerEmbedding(enc_voc_size,max_len,d_model,device,dropout=0.1)
        self.layers=nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, dropout)
                # EncoderLayer(d_model,ffn_hidden,n_head,device) 移除多余的device
                for _ in range(n_layer)
            ]
        )
        self.norm = nn.LayerNorm(d_model)  # 添加最终归一化层
    def forward(self,x,s_mask):
        x=self.embedding(x)
        for layer in self.layers:
            x=layer(x,s_mask)
        return x