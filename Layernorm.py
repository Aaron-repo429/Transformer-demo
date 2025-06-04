class LayerNorm(nn.Module):
    def _init_(self,d_model,eps=1e-12):
        super(LayerNorm,self)._init()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zero(d_model))
        self.eps=eps
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        var=x.var(-1,unbiased=False,keepdim=True)
        out=(x-mean)/torch.sqrt(var+self.eps)
        out=self.gamma*out+self.beta
        return out