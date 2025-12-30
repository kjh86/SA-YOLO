import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[1,2,1],
                          [2,4,2],
                          [1,2,1]], dtype=torch.float32)
        self.register_buffer("k", (k / k.sum()).view(1,1,3,3))

    def forward(self, x):
        c = x.shape[1]
        w = self.k.repeat(c, 1, 1, 1)
        return F.conv2d(x, w, stride=1, padding=1, groups=c)

class AAConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()
        assert s == 2, "AAConv is intended for stride=2 downsampling."
        self.blur = Blur()
        # ✅ autopad 강제: p=None
        self.conv = Conv(c1, c2, k, s, p=None, g=g, d=d, act=act)

    def forward(self, x):
        return self.conv(self.blur(x))
