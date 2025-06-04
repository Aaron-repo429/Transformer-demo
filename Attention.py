import torch
from torch import nn

x=torch.rand(128,32,512)
d_model=512
n_head=8

import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)
        out = self.w_combine(score)
        return out

# 定义模型的维度和头数
d_model = 512
n_head = 8

# 创建多头注意力实例
attention = MultiHeadAttention(d_model, n_head)

out=attention(x,x,x)
print(out)

tensor([[[-0.0490,  0.0948,  0.1095,  ...,  0.1291,  0.2857,  0.3525],
         [-0.0493,  0.0957,  0.1098,  ...,  0.1290,  0.2850,  0.3530],
         [-0.0493,  0.0940,  0.1099,  ...,  0.1286,  0.2857,  0.3524],
         ...,
         [-0.0496,  0.0955,  0.1101,  ...,  0.1289,  0.2863,  0.3530],
         [-0.0491,  0.0937,  0.1101,  ...,  0.1287,  0.2851,  0.3524],
         [-0.0489,  0.0942,  0.1093,  ...,  0.1300,  0.2864,  0.3527]],

        [[-0.0634,  0.1326,  0.1045,  ...,  0.1487,  0.2822,  0.3802],
         [-0.0635,  0.1312,  0.1038,  ...,  0.1496,  0.2823,  0.3798],
         [-0.0632,  0.1319,  0.1037,  ...,  0.1479,  0.2820,  0.3808],
         ...,
         [-0.0645,  0.1320,  0.1048,  ...,  0.1484,  0.2825,  0.3810],
         [-0.0640,  0.1320,  0.1045,  ...,  0.1483,  0.2827,  0.3810],
         [-0.0628,  0.1318,  0.1039,  ...,  0.1497,  0.2821,  0.3804]],

        [[-0.0686,  0.1075,  0.0883,  ...,  0.1295,  0.2744,  0.3690],
         [-0.0690,  0.1077,  0.0880,  ...,  0.1308,  0.2748,  0.3690],
         [-0.0696,  0.1071,  0.0889,  ...,  0.1294,  0.2738,  0.3684],
         ...,
         [-0.0685,  0.1075,  0.0876,  ...,  0.1304,  0.2745,  0.3701],
         [-0.0690,  0.1070,  0.0886,  ...,  0.1300,  0.2743,  0.3686],
         [-0.0697,  0.1068,  0.0891,  ...,  0.1303,  0.2754,  0.3692]],

        ...,
...
         [-0.0661,  0.1054,  0.0837,  ...,  0.1544,  0.2945,  0.3730],
         [-0.0655,  0.1055,  0.0840,  ...,  0.1543,  0.2941,  0.3744],
         [-0.0645,  0.1055,  0.0844,  ...,  0.1550,  0.2944,  0.3741]]],
       grad_fn=<ViewBackward0>)
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
