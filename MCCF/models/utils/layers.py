import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)


def PatchMerging2D(dim, norm_layer=nn.LayerNorm):
    return nn.Sequential(
        Permute(0, 3, 1, 2),
        nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2),
        Permute(0, 2, 3, 1),
        norm_layer(2 * dim),
    )


def PatchExpand2D(dim, norm_layer=nn.LayerNorm):
    return nn.Sequential(
        nn.Linear(dim, 2 * dim, bias=False),
        nn.SiLU(),
        nn.Linear(2 * dim, dim, bias=False),
        norm_layer(dim),
    )


def Final_PatchExpand2D(dim, dim_scale=4, norm_layer=nn.LayerNorm):
    return nn.Sequential(
        nn.Linear(dim, dim * dim_scale, bias=False),
        nn.SiLU(),
        nn.Linear(dim * dim_scale, dim, bias=False),
        norm_layer(dim),
    )
