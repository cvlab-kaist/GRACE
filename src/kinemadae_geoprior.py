# [Source: Wan VAE] wan/modules/vae.py
# [modified] Added extra encoder/decoder stages for higher compression.
# [skip] Skip connections on add_stages via channel averaging (DC-AE style).
# [geoprior] Dual-branch: upper (trainable) + lower (frozen prior encoder on 2x subsampled input).
#   Lower branch z is channel-concatenated with upper branch z before decoder.
#   decoder.conv1 expanded to accept (z_dim + prior_z_dim) input (pretrained in first prior_z_dim ch, zero rest).
#   Supports asymmetric z_dim: e.g. main z_dim=32, prior_z_dim=16 (Wan fixed) → decoder input 48ch.
#   subsample_mode: 'avg_pool' (default) | 'stride' | 'bilinear' — ablation-friendly.
# All modifications are marked with [new] or [modified].
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

__all__ = [
    'WanVAE',
]

CACHE_T = 2


# [skip] 3D pixel shuffle/unshuffle — from Open-Sora dc_ae/models/nn/vo_ops.py
def pixel_shuffle_3d(x, upscale_factor):
    B, C, T, H, W = x.shape
    r = upscale_factor
    assert C % (r * r * r) == 0
    C_new = C // (r * r * r)
    x = x.view(B, C_new, r, r, r, T, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(B, C_new, T * r, H * r, W * r)


def pixel_unshuffle_3d(x, downsample_factor):
    B, C, T, H, W = x.shape
    r = downsample_factor
    assert T % r == 0 and H % r == 0 and W % r == 0
    T_new, H_new, W_new = T // r, H // r, W // r
    x = x.view(B, C, T_new, r, H_new, r, W_new, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(B, C * r * r * r, T_new, H_new, W_new)


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d',
                        'downsample_temporal', 'upsample_temporal')  # [new]
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        # upsample only along time - spatial size and channels unchanged
        elif mode == 'upsample_temporal':
            self.resample = nn.Identity()
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        # downsample only along time - spatial size and channels unchanged
        elif mode == 'downsample_temporal':
            self.resample = nn.Identity()
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        # upsample_temporal shares the temporal logic with upsample3d
        if self.mode in ('upsample3d', 'upsample_temporal'):
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
            else:
                # [PORT from Bfix 2026-06-30] Single-pass temporal upsample: first frame
                #   stored without time_conv (matches chunked first-chunk 'Rep'); remaining
                #   frames upsampled via time_conv -> T_out = 1 + (T_in-1)*2. WITHOUT this
                #   else-branch single-pass left T unchanged -> add_before_head skip mismatch.
                first_out = x[:, :, :1, :, :]
                rest = x[:, :, 1:, :, :]
                if rest.shape[2] > 0:
                    rest_conv = self.time_conv(rest)
                    t_rest = rest.shape[2]
                    rest_conv = rest_conv.reshape(b, 2, c, t_rest, h, w)
                    rest_conv = torch.stack(
                        (rest_conv[:, 0], rest_conv[:, 1]), dim=3)
                    rest_conv = rest_conv.reshape(b, c, t_rest * 2, h, w)
                    x = torch.cat([first_out, rest_conv], dim=2)
                else:
                    x = first_out
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)  # upsample_temporal/downsample_temporal: nn.Identity()
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        # downsample_temporal shares the temporal logic with downsample3d
        if self.mode in ('downsample3d', 'downsample_temporal'):
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
            else:
                # [PORT from Bfix 2026-06-30] Single-pass temporal downsample: first frame
                #   kept as-is; time_conv (padding=(0,0,0)) over full seq matches chunked
                #   per-chunk prepend behaviour. Missing else-branch left encoder T wrong.
                x = torch.cat([x[:, :, :1, :, :], self.time_conv(x)], dim=2)
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def _forward(self, x):
        h = self.shortcut(x)
        for layer in self.residual:
            x = layer(x)
        return x + h

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is None and torch.is_grad_enabled():
            return checkpoint(self._forward, x, use_reentrant=True)
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0,
                 add_stages=None):  # [new] list of {'mode': str, 'num_res_blocks': int}
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # [new] Added downsample stages between downsamples and middle.
        self.add_downsamples = nn.ModuleList()
        if add_stages:
            for stage_cfg in add_stages:
                layers = []
                for _ in range(stage_cfg.get('num_res_blocks', 2)):
                    layers.append(ResidualBlock(out_dim, out_dim, dropout))
                resample = Resample(out_dim, mode=stage_cfg['mode'])
                init_mode = stage_cfg.get('init', 'default')
                if init_mode == 'zero':
                    for p in resample.parameters():
                        nn.init.zeros_(p)
                elif init_mode == 'wan' and hasattr(resample, 'time_conv'):
                    resample.init_weight(resample.time_conv)
                elif init_mode == 'pretrained_copy':
                    src_resblocks = [m for m in self.downsamples if isinstance(m, ResidualBlock) and m.in_dim == out_dim and m.out_dim == out_dim]
                    src_resamples = [m for m in self.downsamples if isinstance(m, Resample) and 'downsample' in m.mode]
                    n_blocks = stage_cfg.get('num_res_blocks', 2)
                    for j in range(min(n_blocks, len(src_resblocks))):
                        layers[j].load_state_dict(src_resblocks[-(n_blocks - j)].state_dict())
                    if src_resamples and resample.mode == src_resamples[-1].mode:
                        resample.load_state_dict(src_resamples[-1].state_dict())
                layers.append(resample)
                self.add_downsamples.append(nn.Sequential(*layers))

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## [skip] added downsample stages with skip connection
        for stage in self.add_downsamples:
            x_in = x
            B, C, T_in, H, W = x_in.shape
            for layer in stage:
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

            resample_mode = next((l.mode for l in stage if isinstance(l, Resample)), 'none')
            if resample_mode == 'downsample3d':
                if T_in == 1:
                    skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                    skip = F.pixel_unshuffle(skip, 2)
                    skip = skip.view(B * T_in, C, 4, H // 2, W // 2).mean(dim=2)
                    skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
                else:
                    skip = pixel_unshuffle_3d(x_in, 2)
                    skip = skip.view(B, C, 8, T_in // 2, H // 2, W // 2).mean(dim=2)
            elif resample_mode == 'downsample2d':
                skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                skip = F.pixel_unshuffle(skip, 2)
                skip = skip.view(B * T_in, C, 4, H // 2, W // 2).mean(dim=2)
                skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
            elif resample_mode == 'downsample_temporal':
                skip = x_in.reshape(B, C, T_in // 2, 2, H, W).mean(dim=3)
            else:
                skip = x_in
            x = x + skip

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0,
                 add_stages=None,
                 add_tail_stages=None,  # [geoprior] stages after head (3ch RGB space)
                 add_before_head_stages=None,  # [geoprior] upsample stages just before head (feature space)
                 dual_branch=False,   # [geoprior]
                 prior_z_dim=None,    # [geoprior] None → same as z_dim (symmetric)
                 expand_conv2=True):  # [geoprior] True: conv2 z_dim→z_dim → decoder input z_dim+prior_z_dim
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # [geoprior] dual_branch decoder input channels:
        #   expand_conv2=True:  conv2(z_dim→z_dim) + conv2_prior(prior_z_dim→prior_z_dim) → z_dim+prior_z_dim
        #   expand_conv2=False: conv2(z_dim→prior_z_dim) + conv2_prior(prior_z_dim→prior_z_dim) → prior_z_dim*2
        if dual_branch:
            _prior_z = prior_z_dim if prior_z_dim is not None else z_dim
            in_z = (z_dim + _prior_z) if expand_conv2 else (_prior_z * 2)
        else:
            in_z = z_dim
        self.conv1 = CausalConv3d(in_z, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # [new] Added upsample stages between middle and upsamples.
        inner_dim = dims[0]
        self.add_upsamples = nn.ModuleList()
        if add_stages:
            for stage_cfg in add_stages:
                layers = []
                mode = stage_cfg['mode']
                n_blocks = stage_cfg.get('num_res_blocks', 2)
                if mode == 'upsample_temporal':
                    for _ in range(n_blocks):
                        layers.append(ResidualBlock(inner_dim, inner_dim, dropout))
                    resample = Resample(inner_dim, mode=mode)
                else:
                    expanded_dim = inner_dim * 2
                    for j in range(n_blocks):
                        if j == 0:
                            layers.append(ResidualBlock(inner_dim, expanded_dim, dropout))
                        else:
                            layers.append(ResidualBlock(expanded_dim, expanded_dim, dropout))
                    resample = Resample(expanded_dim, mode=mode)
                init_mode = stage_cfg.get('init', 'default')
                if init_mode == 'zero':
                    for p in resample.parameters():
                        nn.init.zeros_(p)
                elif init_mode == 'wan' and hasattr(resample, 'time_conv'):
                    resample.init_weight2(resample.time_conv)
                elif init_mode == 'pretrained_copy':
                    resample._deferred_pretrained_copy = True
                layers.append(resample)
                self.add_upsamples.append(nn.Sequential(*layers))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # [geoprior] add_before_head: DC-AE style upsample just before head (out_dim feature space)
        # out_dim here = last upsamples output dim (e.g. 96 for Wan VAE with dim=96)
        # Default init='zero' → step 0 stage output=0, x = pixel_shuffle skip (identity 2× upsample)
        before_head_dim = out_dim
        self.add_before_head = nn.ModuleList()
        if add_before_head_stages:
            for stage_cfg in add_before_head_stages:
                layers = []
                mode = stage_cfg['mode']
                n_blocks = stage_cfg.get('num_res_blocks', 2)
                if mode == 'upsample_temporal':
                    for _ in range(n_blocks):
                        layers.append(ResidualBlock(before_head_dim, before_head_dim, dropout))
                    resample = Resample(before_head_dim, mode=mode)
                else:
                    expanded_dim = before_head_dim * 2
                    for j in range(n_blocks):
                        if j == 0:
                            layers.append(ResidualBlock(before_head_dim, expanded_dim, dropout))
                        else:
                            layers.append(ResidualBlock(expanded_dim, expanded_dim, dropout))
                    resample = Resample(expanded_dim, mode=mode)
                init_mode = stage_cfg.get('init', 'zero')
                if init_mode == 'zero':
                    for p in resample.parameters():
                        nn.init.zeros_(p)
                elif init_mode == 'wan' and hasattr(resample, 'time_conv'):
                    resample.init_weight2(resample.time_conv)
                layers.append(resample)
                self.add_before_head.append(nn.Sequential(*layers))

        # [new] Deferred pretrained_copy init for add_upsamples
        src_resamples = [m for m in self.upsamples if isinstance(m, Resample) and 'upsample' in m.mode]
        for stage in self.add_upsamples:
            for layer in stage:
                if isinstance(layer, Resample) and getattr(layer, '_deferred_pretrained_copy', False):
                    if src_resamples and layer.mode == src_resamples[0].mode:
                        try:
                            layer.load_state_dict(src_resamples[0].state_dict())
                        except RuntimeError:
                            pass
                    if hasattr(layer, '_deferred_pretrained_copy'):
                        del layer._deferred_pretrained_copy

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

        # [geoprior] add_tail: stages after head in 3ch RGB space
        # DC-AE style: zero-init 1x1 proj at end → step 0 output = 0 → x = 0 + x_in = x_in (identity)
        self.add_tail = nn.ModuleList()
        if add_tail_stages:
            for stage_cfg in add_tail_stages:
                layers = []
                for _ in range(stage_cfg.get('num_res_blocks', 2)):
                    layers.append(ResidualBlock(3, 3, dropout))
                proj = CausalConv3d(3, 3, 1)
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                layers.append(proj)
                self.add_tail.append(nn.Sequential(*layers))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## [skip] added upsample stages with skip connection
        for stage in self.add_upsamples:
            x_in = x
            B, C, T_in, H, W = x_in.shape
            for layer in stage:
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

            resample_mode = next((l.mode for l in stage if isinstance(l, Resample)), 'none')
            # Branch on feat_cache (chunked) versus single pass so the skip length matches main exactly,
            #   which removes the _T_min truncate that was losing frames.
            if resample_mode == 'upsample3d':
                if feat_cache is not None:
                    if T_in == 1:
                        skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                        skip = skip.repeat_interleave(4, dim=1)
                        skip = F.pixel_shuffle(skip, 2)
                        skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
                    else:
                        skip = x_in.repeat_interleave(8, dim=1)
                        skip = pixel_shuffle_3d(skip, 2)
                else:
                    skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                    skip = skip.repeat_interleave(4, dim=1)
                    skip = F.pixel_shuffle(skip, 2)
                    skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
                    if T_in > 1:
                        skip = torch.cat([
                            skip[:, :, :1, :, :],
                            skip[:, :, 1:, :, :].repeat_interleave(2, dim=2)
                        ], dim=2)
            elif resample_mode == 'upsample2d':
                skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                skip = skip.repeat_interleave(4, dim=1)
                skip = F.pixel_shuffle(skip, 2)
                skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
            elif resample_mode == 'upsample_temporal':
                if feat_cache is not None:
                    skip = x_in.repeat_interleave(2, dim=2)
                else:
                    if T_in > 1:
                        skip = torch.cat([
                            x_in[:, :, :1, :, :],
                            x_in[:, :, 1:, :, :].repeat_interleave(2, dim=2)
                        ], dim=2)
                    else:
                        skip = x_in.repeat_interleave(2, dim=2)
            else:
                skip = x_in
            x = x + skip

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## [geoprior] add_before_head: DC-AE style upsample in feature space (before head)
        for stage in self.add_before_head:
            x_in = x
            B, C, T_in, H, W = x_in.shape
            for layer in stage:
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)
            resample_mode = next((l.mode for l in stage if isinstance(l, Resample)), 'none')
            # [frame-drop root cause] In chunked decode the main Resample produces a full 2x temporal (2t) per
            #   chunk. Inference had no chunk branch and applied the single-pass logic to the chunked path too,
            #   so it was off by one frame and the _T_min truncate silently dropped frames (17->15, 81->71).
            #   Adding the feat_cache branch and removing the truncate makes the lengths match exactly.
            if resample_mode == 'upsample3d':
                if feat_cache is not None:
                    if T_in == 1:
                        # first chunk: spatial only, 2x
                        skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                        skip = skip.repeat_interleave(4, dim=1)
                        skip = F.pixel_shuffle(skip, 2)
                        skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
                    else:
                        # subsequent chunk: full 2x temporal(2t)
                        skip = x_in.repeat_interleave(8, dim=1)
                        skip = pixel_shuffle_3d(skip, 2)
                else:
                    skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                    skip = skip.repeat_interleave(4, dim=1)
                    skip = F.pixel_shuffle(skip, 2)
                    skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
                    if T_in > 1:
                        skip = torch.cat([
                            skip[:, :, :1, :, :],
                            skip[:, :, 1:, :, :].repeat_interleave(2, dim=2)
                        ], dim=2)
            elif resample_mode == 'upsample2d':
                skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                skip = skip.repeat_interleave(4, dim=1)
                skip = F.pixel_shuffle(skip, 2)
                skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
            elif resample_mode == 'upsample_temporal':
                if feat_cache is not None:
                    if T_in == 1:
                        skip = x_in
                    else:
                        skip = x_in.repeat_interleave(2, dim=2)
                elif T_in > 1:
                    skip = torch.cat([
                        x_in[:, :, :1, :, :],
                        x_in[:, :, 1:, :, :].repeat_interleave(2, dim=2)
                    ], dim=2)
                else:
                    skip = x_in.repeat_interleave(2, dim=2)
            else:
                skip = x_in
            x = x + skip

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        ## [geoprior] add_tail: DC-AE style refinement in 3ch RGB space
        for stage in self.add_tail:
            x_in = x
            for layer in stage:
                if isinstance(layer, ResidualBlock) and feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                elif isinstance(layer, CausalConv3d) and feat_cache is not None:
                    idx = feat_idx[0]
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x], dim=2)
                    x = layer(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                else:
                    x = layer(x)
            x = x + x_in  # DC-AE skip (identity: same resolution, no pixel_shuffle)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0,
                 add_encoder_stages=None,
                 add_decoder_stages=None,
                 add_decoder_tail_stages=None,          # [geoprior] stages after head (3ch RGB space)
                 add_decoder_before_head_stages=None,   # [geoprior] upsample stages just before head
                 dual_branch=False,           # [geoprior]
                 subsample_mode='avg_pool',   # [geoprior] 'avg_pool' | 'stride' | 'bilinear'
                 prior_z_dim=None,            # [geoprior] None → same as z_dim; int for asymmetric (e.g. 16 for frozen Wan)
                 expand_conv2=True,           # [geoprior] True: conv2 z_dim→z_dim; False: conv2 z_dim→prior_z_dim (old)
                 expand_encoder_head=False):  # [geoprior] True: encoder.head outputs z_dim*2 (instead of prior_z_dim*2)
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.dual_branch = dual_branch
        self.subsample_mode = subsample_mode
        # [geoprior] prior branch z_dim (Wan fixed = 16); None → symmetric
        self.prior_z_dim = prior_z_dim if prior_z_dim is not None else z_dim

        self.expand_encoder_head = expand_encoder_head
        # modules
        # [geoprior] dual_branch: encoder head fixed at prior_z_dim*2 (same as pretrained Wan)
        # → encoder.head[-1] keeps shape (384→prior_z_dim*2), no weight mismatch
        # expand_encoder_head=True: encoder.head outputs z_dim*2 (true channel expansion through head)
        enc_out_dim = (self.prior_z_dim * 2) if (dual_branch and not expand_encoder_head) else (z_dim * 2)
        self.encoder = Encoder3d(dim, enc_out_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout,
                                 add_stages=add_encoder_stages)
        # conv1: asymmetric (prior_z_dim*2 → z_dim*2) when dual_branch, square otherwise
        self.conv1 = CausalConv3d(enc_out_dim, z_dim * 2, 1)
        # conv2: expand_conv2=True → z_dim→z_dim; False → z_dim→prior_z_dim (old behavior)
        # decoder.conv1 input: expand_conv2=True → z_dim+prior_z_dim; False → prior_z_dim*2
        self.expand_conv2 = expand_conv2
        if dual_branch:
            _conv2_out = z_dim if expand_conv2 else self.prior_z_dim
            self.conv2 = CausalConv3d(z_dim, _conv2_out, 1)
            if z_dim != self.prior_z_dim:
                # [geoprior] separate frozen conv2 for prior branch
                self.conv2_prior = CausalConv3d(self.prior_z_dim, self.prior_z_dim, 1)
        else:
            self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout,
                                 add_stages=add_decoder_stages,
                                 add_tail_stages=add_decoder_tail_stages,
                                 add_before_head_stages=add_decoder_before_head_stages,  # [geoprior]
                                 dual_branch=dual_branch,
                                 prior_z_dim=self.prior_z_dim,   # [geoprior]
                                 expand_conv2=expand_conv2)       # [geoprior]

        # [geoprior] lower branch: the vanilla Wan encoder (no add_stages, frozen)
        # prior branch uses prior_z_dim (Wan's fixed z_dim=16); weights copied in _video_vae_geoprior
        if dual_branch:
            self.prior_encoder = Encoder3d(dim, self.prior_z_dim * 2, dim_mult, num_res_blocks,
                                           attn_scales, self.temperal_downsample, dropout,
                                           add_stages=None)
            # separate projection for prior branch (conv1 handles main encoder only)
            self.prior_conv1 = CausalConv3d(self.prior_z_dim * 2, self.prior_z_dim * 2, 1)

        # [new] cache count_conv3d results — avoids full module-tree traversal every step
        self._cached_dec_conv_num = count_conv3d(self.decoder)
        self._cached_enc_conv_num = count_conv3d(self.encoder)
        if dual_branch:
            self._cached_prior_conv_num = count_conv3d(self.prior_encoder)

    def forward(self, x):
        mu, log_var = self.encode(x, scale=None)
        z = self.reparameterize(mu, log_var)
        if self.dual_branch:
            with torch.no_grad():
                z_prior = self._encode_prior(x)
            z = torch.cat([z, z_prior], dim=1)  # (B, 2*z_dim, T', H', W')
        x_recon = self.decode(z, scale=None)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        t = x.shape[2]
        tf = 1
        for td in self.temperal_downsample:
            if td:
                tf *= 2
        for stage in self.encoder.add_downsamples:
            for layer in stage:
                if isinstance(layer, Resample) and layer.mode in ('downsample3d', 'downsample_temporal'):
                    tf *= 2
        iter_ = 1 + (t - 1) // tf
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + tf * (i - 1):1 + tf * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu, log_var

    # [geoprior] encode the lower branch
    # depending on subsample_mode, downsample T, H and W by 2, then the frozen prior_encoder gives mu_prior
    def _encode_prior(self, x):
        if self.subsample_mode == 'avg_pool':
            # the causal VAE handles the first frame separately, so T must be odd for the latent T to line up
            # avg_pool3d produces floor(T/2), so when T is odd the last frame is repeated to make it even
            if x.shape[2] % 2 == 1:
                x_pad = torch.cat([x, x[:, :, -1:, :, :]], dim=2)
            else:
                x_pad = x
            x_sub = F.avg_pool3d(x_pad, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        elif self.subsample_mode == 'spatial_avg_temporal_stride':
            # [new] spatial 2x avg + temporal stride 2 (no temporal averaging)
            # temporal averaging is dropped to preserve the I2V first-frame values; only half of the GT frame is used
            # when T is odd the last frame is repeated to make it even (the output T matches 'avg_pool')
            if x.shape[2] % 2 == 1:
                x_pad = torch.cat([x, x[:, :, -1:, :, :]], dim=2)
            else:
                x_pad = x
            x_sub = F.avg_pool3d(x_pad, kernel_size=(1, 2, 2), stride=(2, 2, 2))
        elif self.subsample_mode == 'stride':
            x_sub = x[:, :, ::2, ::2, ::2]
        elif self.subsample_mode == 'bilinear':
            B, C, T, H, W = x.shape
            x_t = x[:, :, ::2, :, :]  # temporal stride
            x_flat = rearrange(x_t, 'b c t h w -> (b t) c h w')
            x_flat = F.interpolate(x_flat, scale_factor=0.5, mode='bilinear', align_corners=False)
            x_sub = rearrange(x_flat, '(b t) c h w -> b c t h w', b=B)
        else:
            raise ValueError(f'Unknown subsample_mode: {self.subsample_mode}')

        t = x_sub.shape[2]
        # prior_encoder has no add_downsamples, so only the base tf applies
        tf = 1
        for td in self.temperal_downsample:
            if td:
                tf *= 2
        prior_feat_map = [None] * self._cached_prior_conv_num
        iter_ = 1 + (t - 1) // tf
        for i in range(iter_):
            prior_idx = [0]
            if i == 0:
                out = self.prior_encoder(
                    x_sub[:, :, :1, :, :],
                    feat_cache=prior_feat_map,
                    feat_idx=prior_idx)
            else:
                out_ = self.prior_encoder(
                    x_sub[:, :, 1 + tf * (i - 1):1 + tf * i, :, :],
                    feat_cache=prior_feat_map,
                    feat_idx=prior_idx)
                out = torch.cat([out, out_], dim=2)
        mu_prior, _ = self.prior_conv1(out).chunk(2, dim=1)
        return mu_prior

    def decode(self, z, scale):
        self.clear_cache()
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        # [geoprior] dual_branch: conv2 for z_main; conv2_prior (frozen) for z_prior
        if self.dual_branch:
            z_main = self.conv2(z[:, :self.z_dim])              # (B, prior_z_dim, T', H', W')
            if self.z_dim == self.prior_z_dim:
                z_p = self.conv2(z[:, self.z_dim:])             # shared conv2
            else:
                z_p = self.conv2_prior(z[:, self.z_dim:])       # separate frozen conv2_prior
            x = torch.cat([z_main, z_p], dim=1)                 # (B, prior_z_dim*2, T', H', W')
        else:
            x = self.conv2(z)
        # force_single_pass: decode in a single pass even in eval mode, avoiding the 81-frame chunk frame-drop
        #   and the T mismatch between tiles. The default is False, which keeps the chunked behaviour.
        if self.training or getattr(self, 'force_single_pass', False):
            out = self.decoder(x)
        else:
            for i in range(iter_):
                self._conv_idx = [0]
                if i == 0:
                    out = self.decoder(
                        x[:, :, i:i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx)
                else:
                    out_ = self.decoder(
                        x[:, :, i:i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx)
                    out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, scale=None, deterministic=False):
        mu, log_var = self.encode(imgs, scale)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = self._cached_dec_conv_num
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = self._cached_enc_conv_num
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu',
               add_encoder_stages=None, add_decoder_stages=None,
               **kwargs):
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        add_encoder_stages=add_encoder_stages,
        add_decoder_stages=add_decoder_stages,
    )
    cfg.update(**kwargs)

    model = WanVAE_(**cfg)

    if pretrained_path is not None:
        logging.info(f'loading {pretrained_path}')
        missing, unexpected = model.load_state_dict(
            torch.load(pretrained_path, map_location=device), strict=False)
        logging.info(f'Loaded: {len(missing)} missing, {len(unexpected)} unexpected keys')

    return model


# [geoprior] dual-branch VAE loader
def _video_vae_geoprior(pretrained_path=None, z_dim=None, device='cpu',
                         add_encoder_stages=None, add_decoder_stages=None,
                         add_decoder_tail_stages=None,          # [NEW] stages after head (3ch RGB space)
                         add_decoder_before_head_stages=None,   # [NEW] upsample stages just before head
                         dual_branch=True, subsample_mode='avg_pool',
                         prior_z_dim=16,  # [NEW] fixed Wan z_dim for prior branch
                         decoder_conv1_zmain_init='zero',  # [NEW] 'zero' or 'pretrained' for z_main ch
                         expand_conv2=True,  # [NEW] True: conv2 z_dim→z_dim; False: conv2 z_dim→prior_z_dim (old)
                         expand_encoder_head=False,  # [NEW] True: encoder.head outputs z_dim*2 (instead of prior_z_dim*2)
                         **kwargs):
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        add_encoder_stages=add_encoder_stages,
        add_decoder_stages=add_decoder_stages,
        add_decoder_tail_stages=add_decoder_tail_stages,
        add_decoder_before_head_stages=add_decoder_before_head_stages,
        dual_branch=dual_branch,
        subsample_mode=subsample_mode,
        prior_z_dim=prior_z_dim if dual_branch else None,
        expand_conv2=expand_conv2,
        expand_encoder_head=expand_encoder_head,
    )
    cfg.update(**kwargs)
    model = WanVAE_(**cfg)

    if pretrained_path is not None:
        logging.info(f'loading {pretrained_path}')
        state = torch.load(pretrained_path, map_location=device)

        # z_dim-dependent keys that have shape mismatch → pop before load_state_dict
        # (strict=False skips missing keys but still errors on size mismatch)
        # New architecture: encoder.head stays (384→prior_z_dim*2) → same as pretrained, NOT popped
        #                   conv1: (prior_z_dim*2 → z_dim*2) asymmetric → popped
        #                   conv2: (z_dim → prior_z_dim) → popped
        #                   decoder.conv1: (prior_z_dim*2 → 384) vs pretrained (prior_z_dim → 384) → popped
        popped = {}
        _zdim_keys = [
            'conv1.weight', 'conv1.bias',          # mu/log_var projection: output expands
            'conv2.weight', 'conv2.bias',           # decoder input projection: input expands
            'decoder.conv1.weight', 'decoder.conv1.bias',  # decoder first conv: input expands
        ]
        if expand_encoder_head:
            # encoder.head[-1] output expands from prior_z_dim*2 to z_dim*2 → shape mismatch
            _zdim_keys += ['encoder.head.2.weight', 'encoder.head.2.bias']
        for k in _zdim_keys:
            if k in state and state[k].shape != model.state_dict().get(k, state[k]).shape:
                popped[k] = state.pop(k)

        missing, unexpected = model.load_state_dict(state, strict=False)
        logging.info(f'Loaded: {len(missing)} missing, {len(unexpected)} unexpected keys')

        if dual_branch:
            with torch.no_grad():
                # 0) encoder.head[-1] (expand_encoder_head=True):
                #    (prior_z_dim*2→prior_z_dim*2) pretrained → new(z_dim*2→z_dim*2)  [actually input stays 384]
                #    Wait: encoder.head[-1] is CausalConv3d(384→enc_out_dim)
                #    pretrained shape: (prior_z_dim*2, 384, 3,3,3); new shape: (z_dim*2, 384, 3,3,3)
                #    copy: mu_old[0:prior_z_dim] → new[0:prior_z_dim]
                #           logvar_old[prior_z_dim:] → new[z_dim:z_dim+prior_z_dim]; rest zero
                if expand_encoder_head:
                    pre_ehw = popped.get('encoder.head.2.weight')
                    if pre_ehw is not None:
                        half = prior_z_dim
                        new_ehw = torch.zeros_like(model.encoder.head[-1].weight)
                        new_ehw[:half, ...] = pre_ehw[:half, ...]                  # mu rows
                        new_ehw[z_dim:z_dim + half, ...] = pre_ehw[half:, ...]    # logvar rows
                        model.encoder.head[-1].weight.copy_(new_ehw)
                    pre_ehb = popped.get('encoder.head.2.bias')
                    if pre_ehb is not None:
                        half = prior_z_dim
                        new_ehb = torch.zeros_like(model.encoder.head[-1].bias)
                        new_ehb[:half] = pre_ehb[:half]
                        new_ehb[z_dim:z_dim + half] = pre_ehb[half:]
                        model.encoder.head[-1].bias.copy_(new_ehb)
                    # conv1 is now z_dim*2→z_dim*2 (both dims changed) → zero-init
                    nn.init.zeros_(model.conv1.weight)
                    if model.conv1.bias is not None:
                        nn.init.zeros_(model.conv1.bias)

                # 1) conv1 (mu/log_var projection): (prior_z_dim*2 → prior_z_dim*2) → (prior_z_dim*2 → z_dim*2)
                #    input dim UNCHANGED (prior_z_dim*2=32), only output expands (32→64)
                #    pretrained rows: [0:prior_z_dim]=mu_old, [prior_z_dim:prior_z_dim*2]=logvar_old
                #    new layout after chunk(2): mu=[0:z_dim], logvar=[z_dim:z_dim*2]
                #    → mu_old rows go to [0:prior_z_dim], logvar_old rows go to [z_dim:z_dim+prior_z_dim]
                if 'conv1.weight' in popped and popped['conv1.weight'] is not None and not expand_encoder_head:
                    pre_c1w = popped['conv1.weight']    # (prior_z_dim*2, prior_z_dim*2, 1,1,1)
                    half = prior_z_dim                  # = prior_z_dim
                    new_c1w = torch.zeros_like(model.conv1.weight)
                    new_c1w[:half, ...] = pre_c1w[:half, ...]                   # mu rows
                    new_c1w[z_dim:z_dim + half, ...] = pre_c1w[half:, ...]     # logvar rows
                    model.conv1.weight.copy_(new_c1w)
                    if 'conv1.bias' in popped and popped['conv1.bias'] is not None:
                        pre_c1b = popped['conv1.bias']
                        new_c1b = torch.zeros_like(model.conv1.bias)
                        new_c1b[:half] = pre_c1b[:half]
                        new_c1b[z_dim:z_dim + half] = pre_c1b[half:]
                        model.conv1.bias.copy_(new_c1b)

                # 2) conv2 (z_main projection):
                #   expand_conv2=True:  pretrained(prior_z_dim→prior_z_dim) → new(z_dim→z_dim), zero-init all
                #   expand_conv2=False: pretrained(prior_z_dim→prior_z_dim) → new(z_dim→prior_z_dim),
                #                       copy pretrained to input ch [0:prior_z_dim], rest zero
                if 'conv2.weight' in popped and popped['conv2.weight'] is not None:
                    if expand_conv2:
                        model.conv2.weight.data.zero_()
                        if model.conv2.bias is not None:
                            model.conv2.bias.data.zero_()
                    else:
                        pre_c2w = popped['conv2.weight']    # (prior_z_dim, prior_z_dim, 1,1,1)
                        new_c2w = torch.zeros_like(model.conv2.weight)
                        new_c2w[:, :prior_z_dim, ...] = pre_c2w
                        model.conv2.weight.copy_(new_c2w)
                        if 'conv2.bias' in popped and popped['conv2.bias'] is not None:
                            model.conv2.bias.copy_(popped['conv2.bias'])

                # 3) decoder.conv1:
                #   expand_conv2=True:  pretrained(prior_z_dim→384) → new(z_dim+prior_z_dim→384)
                #                       z_main ch [0:z_dim]: zero-init; z_prior ch [z_dim:]: pretrained
                #   expand_conv2=False: pretrained(prior_z_dim→384) → new(prior_z_dim*2→384)
                #                       z_prior ch [prior_z_dim:]: pretrained; z_main ch [0:prior_z_dim]: zero or pretrained
                if 'decoder.conv1.weight' in popped and popped['decoder.conv1.weight'] is not None:
                    pre_dc1w = popped['decoder.conv1.weight']   # (384, prior_z_dim, 3,3,3)
                    new_dc1w = torch.zeros_like(model.decoder.conv1.weight)
                    if expand_conv2:
                        # z_main ch [0:z_dim]: stays zero; z_prior ch [z_dim:]: pretrained
                        new_dc1w[:, z_dim:, ...] = pre_dc1w
                    else:
                        if decoder_conv1_zmain_init == 'pretrained':
                            new_dc1w[:, :prior_z_dim, ...] = pre_dc1w   # z_main ch: pretrained copy
                        # else: z_main ch stays zero
                        new_dc1w[:, prior_z_dim:, ...] = pre_dc1w       # z_prior ch: pretrained copy
                    model.decoder.conv1.weight.copy_(new_dc1w)
                    if 'decoder.conv1.bias' in popped and popped['decoder.conv1.bias'] is not None:
                        model.decoder.conv1.bias.copy_(popped['decoder.conv1.bias'])

                # 4) conv2_prior: only when z_dim != prior_z_dim (separate input dim needed)
                #    when z_dim == prior_z_dim: conv2 is shared for both z_main and z_prior
                if z_dim != prior_z_dim:
                    src_c2w = popped.get('conv2.weight')
                    if src_c2w is None:
                        src_c2w = state.get('conv2.weight')
                    if src_c2w is not None and src_c2w.shape == model.conv2_prior.weight.shape:
                        model.conv2_prior.weight.data.copy_(src_c2w)
                    src_c2b = popped.get('conv2.bias') if popped.get('conv2.bias') is not None else state.get('conv2.bias')
                    if src_c2b is not None and model.conv2_prior.bias is not None and src_c2b.shape == model.conv2_prior.bias.shape:
                        model.conv2_prior.bias.data.copy_(src_c2b)
                    model.conv2_prior.requires_grad_(False)

            # prior_encoder: copy the pretrained encoder weights, then freeze
            prior_state = {
                k[len('encoder.'):]: v
                for k, v in state.items()
                if k.startswith('encoder.') and not k.startswith('encoder.add_downsamples')
            }
            # with expand_encoder_head=True the encoder.head.2 weight and bias are popped and missing from prior_state
            # prior_encoder.head[-1] always outputs prior_z_dim*2, the same shape as the pretrained original, so it is restored from the popped values
            if expand_encoder_head:
                for sfx in ('weight', 'bias'):
                    k_enc = f'encoder.head.2.{sfx}'
                    k_pri = f'head.2.{sfx}'
                    if k_enc in popped and popped[k_enc] is not None:
                        prior_state[k_pri] = popped[k_enc]
            model.prior_encoder.load_state_dict(prior_state, strict=False)
            model.prior_encoder.requires_grad_(False)

            # prior_conv1: copy pretrained conv1 weights (same shape prior_z_dim*2 → prior_z_dim*2)
            with torch.no_grad():
                src_w = popped.get('conv1.weight') if popped.get('conv1.weight') is not None else state.get('conv1.weight')
                if src_w is not None and src_w.shape == model.prior_conv1.weight.shape:
                    model.prior_conv1.weight.data.copy_(src_w)
                src_b = popped.get('conv1.bias') if popped.get('conv1.bias') is not None else state.get('conv1.bias')
                if src_b is not None and model.prior_conv1.bias is not None and src_b.shape == model.prior_conv1.bias.shape:
                    model.prior_conv1.bias.data.copy_(src_b)
            model.prior_conv1.requires_grad_(False)

            logging.info(f'prior_encoder+prior_conv1+conv2_prior initialized from pretrained and frozen '
                         f'(prior_z_dim={prior_z_dim})')

    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale)[0].float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
