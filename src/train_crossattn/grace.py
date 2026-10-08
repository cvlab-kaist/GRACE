# [Source: Wan VAE] wan/modules/vae.py
# [Modified - oliviaa] Added extra encoder/decoder stages for higher compression.
# All modifications are marked with [NEW - oliviaa] or [Modified - oliviaa].
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
                        'downsample_temporal', 'upsample_temporal')  # [NEW - oliviaa]
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
        # [NEW - oliviaa] temporal만 upsample — spatial 유지, 채널 유지
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
        # [NEW - oliviaa] temporal만 downsample — spatial 유지, 채널 유지
        elif mode == 'downsample_temporal':
            self.resample = nn.Identity()
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        # [Modified - oliviaa] upsample_temporal은 upsample3d와 temporal 로직 동일
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
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)  # upsample_temporal/downsample_temporal: nn.Identity()
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        # [Modified - oliviaa] downsample_temporal은 downsample3d와 temporal 로직 동일
        if self.mode in ('downsample3d', 'downsample_temporal'):
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
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
                 add_stages=None):  # [NEW - oliviaa] list of {'mode': str, 'num_res_blocks': int}
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

        # [NEW - oliviaa] Added downsample stages between downsamples and middle.
        # Each stage: ResBlock(out_dim, out_dim) x N + Resample(out_dim, mode).
        # Downsample preserves channels, so stages can be stacked freely.
        # init 옵션: "default" (PyTorch 기본), "zero" (zero init),
        #           "wan" (Wan 스타일 identity init), "pretrained_copy" (가장 가까운 stage에서 복사)
        self.add_downsamples = nn.ModuleList()
        if add_stages:
            for stage_cfg in add_stages:
                layers = []
                for _ in range(stage_cfg.get('num_res_blocks', 2)):
                    layers.append(ResidualBlock(out_dim, out_dim, dropout))
                resample = Resample(out_dim, mode=stage_cfg['mode'])
                # [NEW - oliviaa] init 옵션
                init_mode = stage_cfg.get('init', 'default')
                if init_mode == 'zero':
                    for p in resample.parameters():
                        nn.init.zeros_(p)
                elif init_mode == 'wan' and hasattr(resample, 'time_conv'):
                    resample.init_weight(resample.time_conv)
                elif init_mode == 'pretrained_copy':
                    # 가장 가까운 pretrained stage에서 weight 복사
                    # encoder downsamples 끝에서 ResBlock(384,384)×2 + Resample(downsample3d) 찾기
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

        ## [NEW - oliviaa] added downsample stages
        for stage in self.add_downsamples:
            for layer in stage:
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

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
                 add_stages=None):  # [NEW - oliviaa] list of {'mode': str, 'num_res_blocks': int}
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

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # [NEW - oliviaa] Added upsample stages between middle and upsamples.
        # upsample3d/2d: Resample이 채널을 절반으로 줄임 (dim -> dim//2)
        #   → 기존 stage0 입력(384ch)과 맞추려면 미리 2배 확장 필요
        #   → ResBlock(dim, dim*2) x N + Resample(dim*2) -> dim*2//2 = dim
        # upsample_temporal: Resample이 채널을 유지함 (dim -> dim)
        #   → 확장 불필요, ResBlock(dim, dim) x N + Resample(dim) -> dim
        # init 옵션: "default" (PyTorch 기본), "zero" (zero init), "wan" (Wan 스타일 identity init)
        inner_dim = dims[0]  # 384 for default config
        self.add_upsamples = nn.ModuleList()
        if add_stages:
            for stage_cfg in add_stages:
                layers = []
                mode = stage_cfg['mode']
                n_blocks = stage_cfg.get('num_res_blocks', 2)
                if mode == 'upsample_temporal':
                    # [NEW - oliviaa] upsample_temporal은 채널 안 줄이므로 확장 불필요
                    for _ in range(n_blocks):
                        layers.append(ResidualBlock(inner_dim, inner_dim, dropout))
                    resample = Resample(inner_dim, mode=mode)
                else:
                    # upsample3d/2d: 채널 절반 보상을 위해 2배로 확장 후 Resample이 줄임
                    expanded_dim = inner_dim * 2
                    for j in range(n_blocks):
                        if j == 0:
                            layers.append(ResidualBlock(inner_dim, expanded_dim, dropout))
                        else:
                            layers.append(ResidualBlock(expanded_dim, expanded_dim, dropout))
                    resample = Resample(expanded_dim, mode=mode)
                # [NEW - oliviaa] init 옵션
                init_mode = stage_cfg.get('init', 'default')
                if init_mode == 'zero':
                    for p in resample.parameters():
                        nn.init.zeros_(p)
                elif init_mode == 'wan' and hasattr(resample, 'time_conv'):
                    resample.init_weight2(resample.time_conv)
                elif init_mode == 'pretrained_copy':
                    # 가장 가까운 pretrained stage에서 Resample weight만 복사
                    # (ResBlock은 채널이 달라서 복사 불가 — 384→768 vs 384→384)
                    # self.upsamples가 아래에서 생성되므로 여기서는 mark만 해두고 나중에 처리
                    resample._deferred_pretrained_copy = True
                layers.append(resample)
                self.add_upsamples.append(nn.Sequential(*layers))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # [NEW - oliviaa] Deferred pretrained_copy init for add_upsamples
        # self.upsamples가 생성된 후에 처리
        # Resample dim이 다를 수 있으므로 (768 vs 384) strict=False로 시도, 실패 시 skip
        src_resamples = [m for m in self.upsamples if isinstance(m, Resample) and 'upsample' in m.mode]
        for stage in self.add_upsamples:
            for layer in stage:
                if isinstance(layer, Resample) and getattr(layer, '_deferred_pretrained_copy', False):
                    if src_resamples and layer.mode == src_resamples[0].mode:
                        try:
                            layer.load_state_dict(src_resamples[0].state_dict())
                        except RuntimeError:
                            # shape mismatch (e.g., 768 vs 384) — Resample copy 불가, skip
                            pass
                    if hasattr(layer, '_deferred_pretrained_copy'):
                        del layer._deferred_pretrained_copy

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

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

        ## [NEW - oliviaa] added upsample stages
        for stage in self.add_upsamples:
            for layer in stage:
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
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
                 add_encoder_stages=None,   # [NEW - oliviaa]
                 add_decoder_stages=None):  # [NEW - oliviaa]
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout,
                                 add_stages=add_encoder_stages)   # [Modified - oliviaa]
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout,
                                 add_stages=add_decoder_stages)   # [Modified - oliviaa]

    # [Modified - oliviaa] forward passes scale=None to skip normalization during training
    def forward(self, x):
        mu, log_var = self.encode(x, scale=None)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, scale=None)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        # [Modified - oliviaa] 원본: 하드코딩 4 (temporal 4x 기준)
        # 변경: add_stages에 downsample3d가 추가되면 chunk 크기도 커져야 함
        # 예: 원본 4x + add 1개 downsample3d = 8x → chunk 크기 8
        tf = 1
        for td in self.temperal_downsample:
            if td:
                tf *= 2
        for stage in self.encoder.add_downsamples:
            for layer in stage:
                # [Modified - oliviaa] downsample_temporal도 temporal 2x 줄임
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
        # [Modified - oliviaa] scale이 None이면 정규화 건너뜀 (학습 시)
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu, log_var  # [Modified - oliviaa] log_var도 반환

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        # [Modified - oliviaa] scale이 None이면 역정규화 건너뜀 (학습 시)
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
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

    # [Modified - oliviaa] scale 인자 추가 — encode 시그니처 변경에 맞춤
    def sample(self, imgs, scale=None, deterministic=False):
        mu, log_var = self.encode(imgs, scale)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu',
               add_encoder_stages=None, add_decoder_stages=None,  # [NEW - oliviaa]
               **kwargs):
    """
    [Source: Wan VAE]
    [Modified - oliviaa] Added add_encoder_stages/add_decoder_stages params,
    strict=False loading for added stage keys.
    """
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        add_encoder_stages=add_encoder_stages,  # [NEW - oliviaa]
        add_decoder_stages=add_decoder_stages,  # [NEW - oliviaa]
    )
    cfg.update(**kwargs)

    model = WanVAE_(**cfg)

    if pretrained_path is not None:
        logging.info(f'loading {pretrained_path}')
        # [Modified - oliviaa] strict=False — added stage keys will be missing in pretrained ckpt
        missing, unexpected = model.load_state_dict(
            torch.load(pretrained_path, map_location=device), strict=False)
        logging.info(f'Loaded: {len(missing)} missing, {len(unexpected)} unexpected keys')

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

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            # [Modified - oliviaa] encode now returns (mu, log_var), 추론 시 mu만 사용
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
