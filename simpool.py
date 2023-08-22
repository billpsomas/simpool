class SimPool(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, use_gamma=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.gamma = torch.tensor([2.0], device='cuda')
        self.beta = nn.Parameter(torch.tensor([0.0], device='cuda'))
        self.eps = torch.tensor([1e-6], device='cuda')

        self.use_gamma = use_gamma

    def forward(self, q, k, v):

        Bq, Nq, Cq = q.shape
        Bk, Nk, Ck = k.shape
        Bv, Nv, Cv = v.shape

        assert Bq == Bk == Bv
        assert Cq == Ck == Cv

        qq = self.wq(q).reshape(Bq, Nq, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)
        kk = self.wk(k).reshape(Bk, Nk, self.num_heads, Ck // self.num_heads).permute(0, 2, 1, 3)
        
        vv = v.reshape(Bv, Nv, self.num_heads, Cv // self.num_heads).permute(0, 2, 1, 3)

        attn = (qq @ kk.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if self.use_gamma:
            x = torch.pow(attn @ torch.pow((vv - vv.min() + self.eps), self.gamma), 1/self.gamma)  + self.beta
        else:
            x = (attn @ vv).transpose(1, 2).reshape(Bq, Nq, Cq)        
        
        return x