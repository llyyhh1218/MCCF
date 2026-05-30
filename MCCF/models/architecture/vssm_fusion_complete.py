import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, Any
import math
from timm.models.layers import DropPath, trunc_normal_
import torch.utils.checkpoint as checkpoint

from ..core.ss2d import VSSBlock, SS2D
from ..fusion.crm import ChannelRectifyModule
from ..fusion.cross_fusion import VSSBlock_Cross_new
from ..utils.layers import (
    PatchEmbed2D,
    PatchMerging2D,
    PatchExpand2D,
    Final_PatchExpand2D,
)


class VSSLayer(nn.Module):  
    def __init__(
            self,
            dim,
            depth,
            d_state=16,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                **kwargs
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up_VSSB(nn.Module):  
    def __init__(
            self,
            dim,
            depth,
            d_state=16,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # 使用VSSBlock作为解码器基本单元
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                **kwargs
            )
            for i in range(depth)])

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class VSSM_Fusion_Complete(nn.Module): 
    def __init__(self, patch_size=4, in_chans=1, num_classes=1000, depths=[2, 2, 9, 2], 
                 depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], 
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, 
                 img_size=[512, 512], **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
            
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        
        # 双分支Patch Embedding
        self.patch_embed1 = PatchEmbed2D(
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.patch_embed2 = PatchEmbed2D(
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        # 空间分辨率
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        
        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up_VSSB(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)
        
        self.CRM_modules = nn.ModuleList()      
        self.ConMB_modules = nn.ModuleList()     
        self.Cross_fusion_modules = nn.ModuleList()  
        
        for cross_layer in range(self.num_layers):
            HW = (self.patches_resolution[0] // (2 ** cross_layer)) * \
                 (self.patches_resolution[1] // (2 ** cross_layer))
                 
            crm_module = ChannelRectifyModule(
                dim=dims[cross_layer], 
                HW=HW, 
                reduction=16
            )
            self.CRM_modules.append(crm_module)
            
            from ..core.ss2d import ConMB_SS2D
            conmb_module = ConMB_SS2D(
                d_model=dims[cross_layer],
                dropout=attn_drop_rate,
                d_state=d_state,
            )
            self.ConMB_modules.append(conmb_module)
            
            cross_fusion_module = VSSBlock_Cross_new(
                hidden_dim=dims[cross_layer],
                drop_path=drop_rate,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
            )
            self.Cross_fusion_modules.append(cross_fusion_module)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features_1(self, x):
        skip_list = []
        x = self.patch_embed1(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
            
        return x, skip_list

    def forward_features_2(self, x):
        skip_list = []
        x = self.patch_embed2(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
            
        return x, skip_list

    def Fusion_network(self, skip_list1, skip_list2):
        fused_skip_list = []
        
        for idx in range(len(skip_list1)):
            skip1 = skip_list1[idx]
            skip2 = skip_list2[idx]
            
            B, H, W, C = skip1.shape
            
            skip1_2d = skip1.permute(0, 3, 1, 2)  # (B, C, H, W)
            skip2_2d = skip2.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            crm_skip1, crm_skip2 = self.CRM_modules[idx](skip1_2d, skip2_2d)
            
            crm_skip1 = crm_skip1.flatten(2).transpose(1, 2)  # (B, H*W, C)
            crm_skip2 = crm_skip2.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            crm_skip1 = crm_skip1.transpose(1, 2).reshape(B, C, H, W).permute(0, 2, 3, 1)
            crm_skip2 = crm_skip2.transpose(1, 2).reshape(B, C, H, W).permute(0, 2, 3, 1)
            
            processed_skip1, processed_skip2 = self.ConMB_modules[idx](crm_skip1, crm_skip2)
            
            fused_skip = self.Cross_fusion_modules[idx](processed_skip1, processed_skip2)
            
            fused_skip_list.append(fused_skip)
            
        return fused_skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # Skip connection
                x = layer_up(x + skip_list[-inx])
        return x

    def forward_final(self, x):
        """最终输出层"""
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward(self, x1, x2):
        x1, skip_list1 = self.forward_features_1(x1)
        x2, skip_list2 = self.forward_features_2(x2)

        x = x1 + x2
        
        skip_list = self.Fusion_network(skip_list1, skip_list2)

        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)

        return x

    def get_model_info(self):
        info = {
            'model_name': 'VSSM_Fusion_Complete',
            'architecture': 'CRM → ConMB_SS2D → VSSM_Fusion_Method → VSSB_Decoder',
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'dims': self.dims,
            'patches_resolution': self.patches_resolution,
            'num_params': sum(p.numel() for p in self.parameters()),
            'num_trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'fusion_stages': {
                'Stage1': 'CRM (Channel Rectify Module)',
                'Stage2': 'ConMB_SS2D (Feature Processing)',
                'Stage3': 'VSSBlock_Cross_new (True Fusion from FusionMamba-main)',
            }
        }
        return info

def create_vssm_tiny(num_classes=2, in_chans=1, img_size=[256, 256]):
    model = VSSM_Fusion_Complete(
        patch_size=4,
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 9, 2, 2],
        dims=[96, 192, 384, 768],
        dims_decoder=[768, 384, 192, 96],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        img_size=img_size,
    )
    return model


def create_vssm_small(num_classes=2, in_chans=1, img_size=[256, 256]):
    model = VSSM_Fusion_Complete(
        patch_size=4,
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[2, 2, 27, 2],
        depths_decoder=[2, 27, 2, 2],
        dims=[96, 192, 384, 768],
        dims_decoder=[768, 384, 192, 96],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        img_size=img_size,
    )
    return model


def create_vssm_base(num_classes=2, in_chans=1, img_size=[256, 256]):
    model = VSSM_Fusion_Complete(
        patch_size=4,
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[2, 2, 27, 2],
        depths_decoder=[2, 27, 2, 2],
        dims=[128, 256, 512, 1024],
        dims_decoder=[1024, 512, 256, 128],
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.6,
        img_size=img_size,
    )
    return model
