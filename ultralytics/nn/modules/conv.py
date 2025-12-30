# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
    
import torchvision.models as models

import torch.nn.functional as F

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "Concat",
    "RepConv",
    "Index",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y        


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input."""

    def __init__(self, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        return x[self.index]



class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        # 입력 값들을 정수로 보장
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        # expansion이 False인 경우 기본값 4로 설정
        expansion = 4 if expansion is False else int(expansion)
        mid_channels = int(in_channels * expansion)
        
        # Depthwise convolution: groups를 in_channels로 설정 (입력 채널 수와 동일)
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, stride=stride, padding=3, groups=in_channels, bias=True
        )
        # LayerNorm: 채널 차원에 적용 (여기서는 [in_channels, 1, 1] 형태)
        self.norm = nn.LayerNorm([in_channels, 1, 1])
        # Pointwise convolution 역할: 1x1 Conv를 Linear로 구현 (입력: in_channels, 출력: mid_channels)
        self.pwconv1 = nn.Linear(in_channels, mid_channels)
        self.act = nn.GELU()
        # 다시 1x1 Linear: mid_channels → out_channels
        self.pwconv2 = nn.Linear(mid_channels, out_channels)
        
        # 만약 in_channels와 out_channels가 다르거나 stride가 1이 아니라면 shortcut에 projection 추가
        if in_channels != out_channels or stride != 1:
            self.shortcut_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut_proj = nn.Identity()
        
        print(f"[DEBUG] ConvNeXtBlock 생성: in_channels={in_channels}, mid_channels={mid_channels}, "
              f"out_channels={out_channels}, stride={stride}, expansion={expansion}")

    def forward(self, x):
        shortcut = self.shortcut_proj(x)
        x = self.dwconv(x)
        x = self.norm(x)
        B, C, H, W = x.shape
        # Linear 레이어 적용을 위해 spatial 차원을 flatten하고 채널 차원을 뒤로 이동
        x = x.flatten(2).transpose(1, 2)  # shape: [B, H*W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # 다시 원래 형태로 변환: [B, C_out, H, W]
        x = x.transpose(1, 2).view(B, -1, H, W)
        return x + shortcut  # residual 연결


class MobileOneBlock(nn.Module):
    """ MobileOne Block - Reparameterizable Convolution Block """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, reparam=False):
        super().__init__()
        self.reparam = reparam  # True: 추론 최적화 모드

        # 훈련 시에는 여러 개의 경로를 사용
        if not self.reparam:
            self.dw_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=groups, bias=False
            )
            self.pw_conv = nn.Conv2d(
                in_channels, out_channels, 1, 1, 0, bias=False
            )
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            # 추론 시에는 하나의 Conv로 병합
            self.reparam_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=groups, bias=True
            )

    def forward(self, x):
        if self.reparam:
            return self.reparam_conv(x)
        else:
            return self.bn(self.dw_conv(x) + self.pw_conv(x))

    def reparameterize(self):
        """ Reparameterization: 여러 개의 Conv를 하나의 Conv로 변환 """
        if self.reparam:
            return
        kernel, bias = self._fuse_bn()
        self.reparam_conv = nn.Conv2d(
            self.dw_conv.in_channels,
            self.dw_conv.out_channels,
            self.dw_conv.kernel_size,
            self.dw_conv.stride,
            self.dw_conv.padding,
            groups=self.dw_conv.groups,
            bias=True
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # 기존 연산 비활성화
        del self.dw_conv, self.pw_conv, self.bn
        self.dw_conv, self.pw_conv, self.bn = None, None,

class C3MobileOne(nn.Module):
    """CSP Bottleneck with MobileOne reparameterization technique."""

    def __init__(self, c1, c2, n=1, shortcut=True, reparam=False):
        super().__init__()
        c_ = int(c2 * 0.5)  # 내부 채널 수 설정 (기본적으로 c2의 절반)
        self.cv1 = Conv(c1, c_, 1, 1)  # 첫 번째 1x1 Conv
        self.cv2 = Conv(c_, c2, 1, 1)  # 두 번째 1x1 Conv
        self.m = nn.Sequential(*(MobileOneBlock(c_, c_, reparam=reparam) for _ in range(n)))  # MobileOne 블록 추가
        self.add = shortcut and c1 == c2  # Residual 연결 여부

    def forward(self, x):
        y = self.m(self.cv1(x))
        return x + self.cv2(y) if self.add else self.cv2(y)

# class ECA(nn.Module):
#     """Efficient Channel Attention (ECA) Module"""

#     def __init__(self, channels, k_size=3):
#         """
#         Args:
#             channels (int): 입력 채널 수
#             k_size (int): 1D conv의 커널 크기 (채널 간 상호작용 범위)
#         """
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)  # [B, C, 1, 1]
#         y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C]
#         y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
#         return x * y.expand_as(x)



# class ECA(nn.Module):
#     """
#     Efficient Channel Attention (ECA)
#     - k = clamp_odd(|log2(C)*gamma + b|, min=3, max=15)
#     - 매우 경량, 채널 간 상호작용을 1D depthwise conv로 근사
#     """
#     def __init__(self, channel: int, gamma: float = 2.0, b: float = 1.0):
#         super().__init__()
#         k = int(abs(math.log2(max(channel, 1)) * gamma + b))
#         if k % 2 == 0:
#             k += 1
#         k = max(3, min(15, k))  # 안정적 범위로 클램프
#         self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # GAP → [B, C, 1, 1] → [B, 1, C] → Conv1d → [B, 1, C] → Sigmoid → [B, C, 1, 1]
#         y = F.adaptive_avg_pool2d(x, 1)                # [B,C,1,1]
#         y = y.squeeze(-1).squeeze(-1).unsqueeze(1)     # [B,1,C]
#         y = self.conv(y).squeeze(1)                    # [B,C]
#         y = torch.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
#         return x * y.expand_as(x)
    

class SPP_CSP(nn.Module):
    """Spatial Pyramid Pooling with CSP connections (SPP_CSP)."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple or list, optional): Kernel sizes for pooling. Defaults to (5, 9, 13).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)

        # ✅ k가 int로 전달되었을 경우 자동 변환
        if isinstance(k, int):
            k = [k]  # 리스트로 변환

        self.cv2 = Conv(c_ * (len(k) + 1), c_, 1, 1)  # ✅ len(k) 사용 가능하도록 수정
        self.cv3 = Conv(c_, c2, 3, 1)  # 최종 Conv

        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP_CSP layer."""
        x = self.cv1(x)
        x = self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        return self.cv3(x)

# class WindowAttBlock(nn.Module):
#     """
#     간단한 예시용 윈도우 기반 Attention/Conv Block:
#     - feature map을 window_size로 나눈 뒤 (unfold)
#     - 작은 window 단위 Conv/Attention 적용
#     - 다시 fold해서 복원
#     """

#     def __init__(self, channels, window_size=8):
#         super().__init__()
#         self.channels = channels
#         self.window_size = window_size
#         # 예: 윈도우 내부 처리를 위해 3x3 Conv
#         self.local_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # 1) unfold로 window_size x window_size 패치 추출
#         #    stride, kernel_size를 window_size로 해서 non-overlapping
#         #    => shape: [B, C * window_size^2, num_windows]
#         #    여기서는 단순예시
#         unfolded = F.unfold(
#             x,
#             kernel_size=(self.window_size, self.window_size),
#             stride=(self.window_size, self.window_size),
#             padding=0
#         )  # [B, C*win^2, (H/win)*(W/win)]

#         # unfolded.shape = [B, C*(win^2), N],  N = #windows

#         # 2) 로컬 Conv or Attention
#         #    여기선 그냥 1D Conv 취급 or reshape→local_conv
#         #    간단화 위해, reshape → local_conv → reshape

#         # unfold된 patch별로 처리하려면 일단 patch를 2D 형상으로 되돌림
#         # patches: [B*N, C, win, win]
#         num_windows = unfolded.shape[-1]
#         patch_hw = self.window_size * self.window_size
#         # (B, C*win^2, N) -> (B, N, C, win^2) -> (B*N, C, win^2)
#         patches = unfolded.transpose(1,2).reshape(-1, C, self.window_size, self.window_size)

#         # 여기에 윈도우 단위 Conv
#         patches_out = self.local_conv(patches)
#         # shape same: [B*N, C, win, win]

#         # 3) 다시 fold
#         #    patches_out -> [B, N, C*win^2], trans -> [B, C*win^2, N]
#         patches_out = patches_out.reshape(-1, self.window_size*self.window_size*C, 1)
#         # patches_out = [B*N, C*win^2, 1],   but we want [B, C*(win^2), N]
#         patches_out = patches_out.view(B, num_windows, -1).transpose(1,2)
#         # => shape [B, C*win^2, N]

#         # fold => [B, C, H, W]
#         #   kernel_size=win, stride=win
#         x_out = F.fold(
#             patches_out,
#             output_size=(H, W),
#             kernel_size=(self.window_size, self.window_size),
#             stride=(self.window_size, self.window_size)
#         )
#         return x_out

class WindowAttBlock(nn.Module):
    """
    간단한 예시용 윈도우 기반 Attention/Conv Block:
    - feature map을 window_size로 나눈 뒤 (unfold)
    - 작은 window 단위 Conv/Attention 적용
    - 다시 fold해서 복원
    """

    def __init__(self, channels, window_size=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        # 예: 윈도우 내부 처리를 위해 3x3 Conv
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        print(f"[DEBUG] local_conv: {self.local_conv}")  # ✅ 채널 확인

        B, C, H, W = x.shape
        
        # 입력 크기가 window_size로 나누어 떨어지는지 확인하고 필요시 패딩
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            # 패딩 후 크기 업데이트
            _, _, H, W = x.shape
        
        # 1) unfold로 window_size x window_size 패치 추출
        #    stride, kernel_size를 window_size로 해서 non-overlapping
        unfolded = F.unfold(
            x,
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size),
            padding=0
        )  # [B, C*win^2, (H/win)*(W/win)]

        # 2) 로컬 Conv or Attention 적용을 위한 reshape
        num_windows = unfolded.shape[-1]  # (H/win)*(W/win)
        
        # [B, C*win^2, num_windows] -> [B, num_windows, C*win^2] -> [B*num_windows, C, win, win]
        patches = unfolded.transpose(1, 2).contiguous().view(
            B * num_windows, C, self.window_size, self.window_size
        )

        # 윈도우 단위 Conv 적용
        patches_out = self.local_conv(patches)  # [B*num_windows, C, win, win]

        # 3) 다시 fold를 위한 reshape
        # [B*num_windows, C, win, win] -> [B, num_windows, C*win^2] -> [B, C*win^2, num_windows]
        patches_out = patches_out.view(
            B, num_windows, C * self.window_size * self.window_size
        ).transpose(1, 2).contiguous()

        # fold로 원래 크기 복원 => [B, C, H, W]
        x_out = F.fold(
            patches_out,
            output_size=(H, W),
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size)
        )
        
        # 패딩을 추가했다면 원래 크기로 다시 잘라내기
        if pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :, :H-pad_h, :W-pad_w]
            
        return x_out

class MultiWindowC2f(nn.Module):
    """
    Multi-Window C2f 모듈 예시:
    - 입력 -> cv1 -> (m개 window block 병렬) -> concat -> cv2 -> (optional) residual
    """

    def __init__(self, c1, c2, n=1, window_sizes=(8, 4), e=0.5, residual=False):
        """
        Args:
            c1 (int): input channels
            c2 (int): output channels
            n  (int): 병렬 block 개수
            window_sizes (tuple): 각 block에 사용할 window_size 목록
            e (float): expansion ratio -> hidden channels
            residual (bool): residual 적용 여부
        """
        super().__init__()
        
        # n과 window_sizes 길이 일치시키기
        if len(window_sizes) != n:
            print(f"Warning: n({n})과 window_sizes({len(window_sizes)})의 길이가 다릅니다. window_sizes의 길이({len(window_sizes)})를 n으로 사용합니다.")
            n = len(window_sizes)

        c_ = int(c2 * e)  # 중간 채널 수

        # 1) cv1: 입력 -> 중간채널
        self.cv1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1)

        # 2) 각 window_size에 대한 병렬 블록
        self.blocks = nn.ModuleList([
            WindowAttBlock(c_, window_size=ws) for ws in window_sizes
        ])

        # 3) cv2: concat -> c2
        #    병렬 branch가 n개 => concat 후 channel = n * c_
        self.cv2 = nn.Conv2d(n*c_, c2, kernel_size=1, stride=1)

        # residual 파라미터
        self.residual = residual
        # 입/출력 채널이 다르면 1x1 conv로 매핑
        self.shortcut = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, kernel_size=1)
        
        if self.residual:
            # scale param (0으로 초기화)
            self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        else:
            self.gamma = None

    def forward(self, x):
        """
        1) cv1 -> y
        2) 병렬 window-att block 처리 -> outputs list
        3) concat channel dim
        4) cv2
        5) optional residual
        """
        identity = self.shortcut(x)  # identity mapping (채널 수 조정 필요 시)
        y = self.cv1(x)  # shape [B, c_, H, W]

        # 각 블록 병렬 처리
        outs = []
        for block in self.blocks:
            out_b = block(y)
            outs.append(out_b)

        # concat along channel => shape [B, n*c_, H, W]
        cat_out = torch.cat(outs, dim=1)
        y_final = self.cv2(cat_out)  # -> [B, c2, H, W]

        if self.residual:
            # residual connection 적용
            return identity + self.gamma * y_final
        else:
            return y_final









class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class PAM(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.eca = ECA(channels, gamma=gamma, b=b)
        self.caca_eca = ResECA(channels, gamma=gamma, b=b)

    def forward(self, x):
        out1 = self.eca(x)                # 단순한 채널 attention
        out2 = self.caca_eca(x)           # 복합 attention + residual
        out = (out1 + out2) / 2           # 평균 (or concat 후 conv 가능)
        return out
    
class ResECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.eca = ECA(channels, gamma=gamma, b=b)  # ✅ adaptive 방식

    def forward(self, x):
        out = self.channel_att(x)
        out = self.eca(out)
        return out + x      



class ECA(nn.Module):
    """Efficient Channel Attention (ECA) module."""

    def __init__(self, channel, gamma=2, b=1):
        """
        ECA Attention Layer
        :param channel: Number of channels in the input feature map.
        :param gamma: Used to calculate the kernel size.
        :param b: Bias term in kernel size calculation.
        """
        super().__init__()
        # Adaptive kernel size: k = log2(C) * gamma + b
        kernel_size = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) * gamma + b).item()))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1  # Make sure it's odd

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        """Applies ECA to the input tensor."""
        y = torch.mean(x, dim=(2, 3), keepdim=True)  # Global Average Pooling (GAP)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 1D Convolution
        y = torch.sigmoid(y)  # Sigmoid activation
        return x * y.expand_as(x)  # Element-wise multiplication    