import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_

from .blocks import Block, FeedForward
from .utils import init_weights


class MaskLayerNorm(nn.Module):
    def __init__(self, num_class, eps=1e-5, elementwise_affine=True):
        super(MaskLayerNorm, self).__init__()
        self.num_class = num_class
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.ParameterList(nn.Parameter(torch.ones(c)) for c in self.num_class)
            self.bias = nn.ParameterList(nn.Parameter(torch.zeros(c)) for c in self.num_class)

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        normed_input = (input - mean) / (std + self.eps)
        
        weight = [weight for weight in self.weight]
        bias = [bias for bias in self.bias]
        
        weight = torch.cat(weight)
        bias = torch.cat(bias)
        
        if self.elementwise_affine:
            normed_input = normed_input * weight + bias
        
        return normed_input
    
class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x

class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.ParameterList(nn.Parameter(torch.randn(1, c, d_model)) for c in self.n_cls)
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.sum_cls = sum(n_cls)
        self.mask_norm = MaskLayerNorm(self.n_cls)
        
        self.patches = None
        self.cls_seg_feat = None
        self.cls_token = None
        
        self.apply(init_weights)
        for param in self.cls_emb:
            trunc_normal_(param, std=.02)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def return_feats(self):
        return self.patches, self.cls_seg_feat, self.cls_token
        
    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.proj_dec(x)
        
        cls_embeds = [cls_embed for cls_embed in self.cls_emb]
        cls_embeds = torch.cat(cls_embeds, dim=1)
        cls_embeds = cls_embeds.expand(x.size(0), -1, -1)
        self.cls_token = cls_embeds
        x = torch.cat((x, cls_embeds), 1)
        for blk in self.blocks:
            x = blk(x)
            
        x = self.decoder_norm(x)
        
        patches, cls_seg_feat = x[:, :-self.sum_cls], x[:, -self.sum_cls:]
        
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        
        self.patches = patches
        self.cls_seg_feat = cls_seg_feat
        
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)
        
        masks = patches @ cls_seg_feat.transpose(1, 2)
        
        masks = self.mask_norm(masks)
    
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        
        return masks
