import torch
import torch.nn as nn
class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0., use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()
        
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        self.f = nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )              
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, C, H, W = x.shape
        x = x.contiguous().view(B, H, W, C)
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level+1), 1)
        self.gates = torch.sigmoid(self.gates)
        
        # context aggreation
        ctx_all = 0 
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = (1 - self.gates[:,self.focal_level:])*ctx_all + ctx_global*(1 + self.gates[:,self.focal_level:])

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level+1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q*self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        x_out = x_out.contiguous().view(B, C, H, W)
        return x_out
    
if __name__ == '__main__':
    x = torch.randn(8, 128, 8, 8)
    model = FocalModulation(128, 3, 3)
    out = model(x)
    
    print(out.shape)