import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, Any
import math
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange, repeat
from ..core.ss2d import SS2D


class ChannelWeights(nn.Module):
    def __init__(self, dim, channel_dim, reduction=4):
        super(ChannelWeights, self).__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 96),
            nn.GELU(),
            SS1D(d_model=96, dropout=0, d_state=16),
            nn.LayerNorm(96),
            nn.Linear(96, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1).view(B, 2*C, H*W)
        x = self.mlp(x).view(B, 2*C, 1)
        channel_weights = x.reshape(B, 2, C, 1, 1).permute(1, 0, 2, 3, 4)
        return channel_weights


class SS1D(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            d_conv=-1,
            conv_bias=True,
            dropout=0.,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            softmax_version=False,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.K = 1
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            SS2D.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.K2 = 1
        self.A_logs = SS2D.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)
        self.Ds = SS2D.D_init(self.d_inner, copies=self.K2, merge=True)

        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward_corev0(self, x: torch.Tensor):
        B, L, HW = x.shape
        y = x.mean(dim=-1, keepdim=True).expand_as(x)
        
        if not self.softmax_version and hasattr(self, 'out_norm'):
            y = self.out_norm(y)
        
        return y

    forward_core = forward_corev0

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)
            x = x.permute(0, 2, 1).contiguous()
            x = x.unsqueeze(-1)
            x = self.act(self.conv2d(x))
            x = x.squeeze(-1)
            y = self.forward_core(x)
            if self.softmax_version:
                y = y * z
            else:
                y = y * F.silu(z)
        else:
            if self.softmax_version:
                x, z = xz.chunk(2, dim=-1)
                x = F.silu(x)
            else:
                xz = F.silu(xz)
                x, z = xz.chunk(2, dim=-1)
            y = self.forward_core(x)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class ChannelRectifyModule(nn.Module):
    def __init__(self, dim, HW, reduction=16, lambda_c=0.5, lambda_s=0.5):
        super(ChannelRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=HW, channel_dim=dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        out_x1 = x1 + channel_weights[0] * x1
        out_x2 = x2 + channel_weights[1] * x2
        return out_x1, out_x2


class ConMB_SS2D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        d_conv=3,
        conv_bias=True,
        dropout=0.,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        softmax_version=False,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_modalx = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.K = 8
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            SS2D.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.K2 = self.K
        self.A_logs = SS2D.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)
        self.Ds = SS2D.D_init(self.d_inner, copies=self.K2, merge=True)

        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner*2, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans))
            x_e_conv = self.act(self.conv2d_modalx(x_e_trans))
            
            b, d, h, w = x_rgb_conv.shape
            
            y_rgb = x_rgb_conv.view(b, d, -1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1).expand_as(x_rgb_conv)
            y_e = x_e_conv.view(b, d, -1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1).expand_as(x_e_conv)
            
            y_rgb = y_rgb.permute(0, 2, 3, 1)
            y_e = y_e.permute(0, 2, 3, 1)
            
            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            
            x_rgb_exitation = self.fc1(x_e_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()
            x_e_exitation = self.fc2(x_rgb_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()
                      
            processed_y_rgb = y_rgb * x_e_exitation
            processed_y_e = y_e * x_rgb_exitation
            
            out = self.dropout(self.out_proj(torch.cat([processed_y_rgb, processed_y_e], dim=-1)))
            
            return x_rgb + processed_y_rgb * 0.5, x_e + processed_y_e * 0.5


class Conv2d_Hori_Veri_Cross(nn.Module):
    """水平垂直方向卷积"""
    def __init__(self, inp, oup, kernel_size=3):
        super().__init__()
        self.hor_conv = nn.Conv2d(inp, oup // 2, (1, kernel_size), 1, (0, kernel_size // 2))
        self.ver_conv = nn.Conv2d(inp, oup // 2, (kernel_size, 1), 1, (kernel_size // 2, 0))

    def forward(self, x):
        return torch.cat([self.hor_conv(x), self.ver_conv(x)], dim=1)


class Conv2d_Diag_Cross(nn.Module):
    def __init__(self, inp, oup, kernel_size=3):
        super().__init__()
        self.dia_conv = nn.Conv2d(inp, oup, kernel_size, 1, kernel_size // 2)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def forward(self, x):
        return self.dia_conv(self.pad(x))


class LDC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LDC, self).__init__()
        self.conv = nn.Sequential(
            Conv2d_Hori_Veri_Cross(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            Conv2d_Diag_Cross(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class Enhancement_texture_LDC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enhancement_texture_LDC, self).__init__()
        self.texture_enhance = LDC(in_channels, out_channels)

    def forward(self, x):
        return self.texture_enhance(x)


class Differential_enhance(nn.Module):
    def __init__(self, dim):
        super(Differential_enhance, self).__init__()
        self.diff_enhance = LDC(dim, dim)

    def forward(self, Fuse, x1, x2):
        diff = torch.abs(x1 - x2)
        DF = self.diff_enhance(diff)
        
        DF_x1 = DF * x1
        DF_x2 = DF * x2
        
        return DF_x1, DF_x2


class Cross_layer(nn.Module):
    def __init__(self, hidden_dim: int = 0):
        super().__init__()
        self.d_model = hidden_dim
        self.texture_enhance1 = Enhancement_texture_LDC(self.d_model, self.d_model)
        self.texture_enhance2 = Enhancement_texture_LDC(self.d_model, self.d_model)
        self.Diff_enhance = Differential_enhance(self.d_model)

    def forward(self, Fuse, x1, x2):
        TX_x1 = self.texture_enhance1(x1)
        TX_x2 = self.texture_enhance2(x2)
        
        DF_x1, DF_x2 = self.Diff_enhance(Fuse, x1, x2)
        
        F_1 = TX_x1 + DF_x1
        F_2 = TX_x2 + DF_x2
        
        return F_1, F_2


class SS2D_cross_new(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv

        self.out_norm = nn.LayerNorm(d_inner)

        self.in_proj1 = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act1: nn.Module = act_layer()
        self.act2: nn.Module = act_layer()

        if d_conv > 1:
            self.conv2d1 = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d2 = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act_conv: nn.Module = act_layer()

        self.K = 4
        self.x_proj1 = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj1_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj1], dim=0))
        del self.x_proj1

        self.x_proj2 = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj2_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj2], dim=0))
        del self.x_proj2

        self.dt_projs1 = [
            SS2D.dt_init(self.dt_rank, d_inner, 1.0, "random", 0.001, 0.1, 1e-4, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs1_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs1], dim=0))
        self.dt_projs1_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs1], dim=0))
        del self.dt_projs1

        self.dt_projs2 = [
            SS2D.dt_init(self.dt_rank, d_inner, 1.0, "random", 0.001, 0.1, 1e-4, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs2_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs2], dim=0))
        self.dt_projs2_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs2], dim=0))
        del self.dt_projs2

        self.K2 = self.K
        self.A_logs1 = SS2D.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)
        self.Ds1 = SS2D.D_init(d_inner, copies=self.K2, merge=True)
        self.A_logs2 = SS2D.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)
        self.Ds2 = SS2D.D_init(d_inner, copies=self.K2, merge=True)

        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward_corev2(self, x1, x2):
        B, C, H, W = x1.shape
        L = H * W
        
        xs1 = torch.cat([
            x1.view(B, -1, L),
            x1.transpose(dim0=2, dim1=3).contiguous().view(B, -1, L),
            torch.flip(x1, dims=[-1]).view(B, -1, L),
            torch.flip(x1.transpose(dim0=2, dim1=3), dims=[-1]).view(B, -1, L)
        ], dim=1)
        
        xs2 = torch.cat([
            x2.view(B, -1, L),
            x2.transpose(dim0=2, dim1=3).contiguous().view(B, -1, L),
            torch.flip(x2, dims=[-1]).view(B, -1, L),
            torch.flip(x2.transpose(dim0=2, dim1=3), dims=[-1]).view(B, -1, L)
        ], dim=1)
        
        y = (xs1[:, :4] + xs2[:, :4]) / 2
        y = y[:, 0]
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        
        return y

    forward_core = forward_corev2

    def forward(self, input1, input2):
        xz1 = self.in_proj1(input1)
        xz2 = self.in_proj2(input2)
        
        x1, z1 = xz1.chunk(2, dim=-1)
        x2, z2 = xz2.chunk(2, dim=-1)
        
        x1 = self.act1(x1)
        x2 = self.act2(x2)
        
        if self.d_conv > 1:
            x1 = x1.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            
            x1 = self.act_conv(self.conv2d1(x1))
            x2 = self.act_conv(self.conv2d2(x2))
            
            y = self.forward_core(x1, x2)
        else:
            x1 = x1.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            y = self.forward_core(x1, x2)
        
        z = (z1 + z2) / 2
        y = y * F.silu(z)
        out = self.dropout(self.out_proj(y))
        
        return out


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class VSSBlock_Cross_new(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        
        self.Cross_layer = Cross_layer(hidden_dim)
        self.self_attention_cross = SS2D_cross_new(
            d_model=hidden_dim, 
            dropout=attn_drop_rate, 
            d_state=d_state, 
            **kwargs
        )
        self.self_attention_cross_spatial = eca_layer(channel=hidden_dim)
        
        self.drop_path = DropPath(drop_path)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        x_1 = input1.permute(0, 3, 1, 2)
        x_2 = input2.permute(0, 3, 1, 2)

        Fuse = torch.add(x_1, x_2, alpha=1)

        F_1, F_2 = self.Cross_layer(Fuse, x_1, x_2)
        
        F_1 = F_1.permute(0, 2, 3, 1)
        F_2 = F_2.permute(0, 2, 3, 1)
        
        Cross_x1x2 = self.self_attention_cross(
            self.ln_1(F_1), 
            self.ln_2(F_2)
        )  # (b, h, w, c)

        Cross_x1x2_ = Cross_x1x2.permute(0, 3, 1, 2)  # (b, c, h, w)
        Cross_x1x2_spatial = self.self_attention_cross_spatial(Cross_x1x2_)
        Cross_x1x2_spatial = Cross_x1x2_spatial.permute(0, 2, 3, 1)  # (b, h, w, c)
        
        x = input2 + input1 + Cross_x1x2 + Cross_x1x2_spatial
        
        return x
