# [Source: Wan VAE] wan/modules/vae.py
# [Modified] Added extra encoder/decoder stages for higher compression.
# [NEW/skip] Skip connections on add_stages via channel averaging (DC-AE style).
# [NEW/geoprior] Dual-branch: upper (trainable) + lower (frozen prior encoder on 2x subsampled input).
#   Lower branch z is channel-concatenated with upper branch z before decoder.
#   decoder.conv1 expanded to accept (z_dim + prior_z_dim) input (pretrained in first prior_z_dim ch, zero rest).
#   Supports asymmetric z_dim: e.g. main z_dim=32, prior_z_dim=16 (Wan fixed) → decoder input 48ch.
#   subsample_mode: 'avg_pool' (default) | 'stride' | 'bilinear' — ablation-friendly.
# All modifications are marked with [NEW] or [Modified].
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import os
# [NEW 2026-07-20] Extends VAE gradient checkpointing. When on, Resample (the up/downsample
#   convs) is checkpointed too; ResidualBlock always is. It only checkpoints on the
#   grad + single-pass path (feat_cache None), which lowers the VAE activation floor and lets
#   num_frames go up. Equivalence was verified in verify_vae_gc_equiv.py: the forward is
#   bit-identical and gradients differ only at floating-point noise level. Off by default.
_EXTENDED_VAE_GC = os.environ.get("GRACE_EXTENDED_VAE_GC", "0") == "1"

__all__ = [
    'WanVAE',
]

CACHE_T = 2


# [NEW/skip] 3D pixel shuffle/unshuffle — from Open-Sora dc_ae/models/nn/vo_ops.py
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
        # 1x1x1 fast path: F.conv3d backward via cuDNN produces weight-grad
        # tensors with non-standard strides ([C_in,1,C_in,C_in,C_in] instead
        # of [C_in,1,1,1,1]), triggering DDP reducer copy-warnings on every
        # step.  F.linear backward always produces a contiguous weight grad
        # (standard GEMM), and also avoids the F.pad no-op copy that the
        # general path makes for k=1 (verified: F.pad with all-zero padding
        # allocates a new tensor).
        if all(k == 1 for k in self.kernel_size):
            B, C_in = x.shape[0], x.shape[1]
            x_flat = x.movedim(1, -1).reshape(-1, C_in)
            out = F.linear(x_flat, self.weight.view(self.out_channels, C_in), self.bias)
            return out.reshape(*x.shape[:1], *x.shape[2:], self.out_channels).movedim(-1, 1)

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
                        'downsample_temporal', 'upsample_temporal',
                        'upsample2d_keepdim')  # [NEW - R2/M1] spatial upsample that keeps the width (upsample2d goes dim -> dim//2)
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        # [NEW - R2/M1] for the decoder mirror: keeps the width (384->384). Zero-init leaves only
        # the skip, a channel-preserving nearest upsample.
        elif mode == 'upsample2d_keepdim':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        # [NEW] upsample time only - spatial size and channel count stay
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
        # [NEW] downsample time only - spatial size and channel count stay
        elif mode == 'downsample_temporal':
            self.resample = nn.Identity()
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # [NEW 2026-07-20] extended checkpointing: only on the grad + single-pass path
        #   (feat_cache None). feat_idx is mutated only in the feat_cache branch, so a
        #   no-cache recompute is safe.
        if _EXTENDED_VAE_GC and feat_cache is None and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, x, feat_cache, feat_idx, use_reentrant=False)
        return self._forward_impl(x, feat_cache, feat_idx)

    def _forward_impl(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        # [Modified] upsample_temporal shares upsample3d's temporal logic
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
                # Single-pass: first frame stored without time_conv (matches chunked
                # first-chunk behaviour); remaining frames upsampled via time_conv.
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

        # [Modified] downsample_temporal shares downsample3d's temporal logic
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
                # Single-pass: first frame stored without time_conv (matches chunked
                # first-chunk behaviour); remaining frames downsampled via time_conv.
                # time_conv has no causal padding (padding=(0,0,0)), so time_conv(x)
                # starting from frame0 produces the same outputs as chunked chunks
                # where each chunk prepends the last cached frame.
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
            return checkpoint(self._forward, x, use_reentrant=False)

        # No-grad / cached path:
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
        # Use F.linear instead of Conv2d(k=1) forward: cuDNN 1x1 backward
        # produces non-standard weight-grad strides, triggering DDP warnings.
        x_flat = x.movedim(1, -1).reshape(-1, c)
        q, k, v = (F.linear(x_flat, self.to_qkv.weight.view(c * 3, c), self.to_qkv.bias)
                   .reshape(b * t, 1, h * w, c * 3).chunk(3, dim=-1))

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).reshape(b * t, h * w, c).movedim(1, -1).reshape(b * t, c, h, w)

        # output
        x_flat = x.movedim(1, -1).reshape(-1, c)
        x = (F.linear(x_flat, self.proj.weight.view(c, c), self.proj.bias)
             .reshape(b * t, h, w, c).movedim(-1, 1))
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
                 add_stages=None,  # [NEW] list of {'mode': str, 'num_res_blocks': int}
                 da_adapter=False,        # [NEW - da_adapter] DA-VAE-style compressor: channel compression (to z_dim/r^2) plus pixel_unshuffle(r), in place of stages
                 da_spatial_fold=4,       # [NEW - da_adapter] spatial fold factor r (4 for f32t4, 2 for f16t4)
                 stages_after_norm=False,  # [NEW - R2] run add_downsamples after middle -> RMS_norm -> SiLU, which puts mid back at full resolution
                 stages_norm_before_head=False,  # [NEW - R2n] re-normalise with RMSnorm between the R2 stack output and the head conv (spike mitigation 1)
                 stages_after_head=False,  # [NEW - R3] run add_downsamples *after* the head conv, at width z_dim, so the head is at full resolution too
                 stages_after_conv1=False,  # [NEW - R3a] only *build* the stages here; GeopriorVAE.encode runs them after conv1
                 use_b_adaptive=False,  # [NEW/B-fix] replace encoder.head[-1] with AdaptiveWeightedCausalConv3d, the single-backward-path weight gradient ratio mechanism
                 b_adaptive_eps=1e-6,
                 b_adaptive_max=1e7,
                 b_adaptive_disc_weight=1.0):
        super().__init__()
        # [NEW/B-fix] keep the flag: forward uses it to decide how to handle the head's two outputs
        self.use_b_adaptive = use_b_adaptive
        # [NEW - da_adapter]
        self.da_adapter = da_adapter
        self.da_fold = da_spatial_fold
        # [NEW - R2]
        self.stages_after_norm = stages_after_norm
        assert not (da_adapter and stages_after_norm), "[R2/da] these structural modes are mutually exclusive"
        # [NEW - R2n] spike mitigation 1. In R2 a free 34.5M stack sits after norm and SiLU, which
        #   removes the normalisation anchor on the head conv's input - the epicentre of the 7th
        #   divergence, where grad_W alone grew 16x. Re-normalising the stack output with RMSnorm
        #   breaks that loop. The norm only ties the scale and leaves the channel arrangement,
        #   and so the alignment freedom, intact. Off by default, which is bit-identical.
        self.stages_norm_before_head = stages_norm_before_head
        if stages_norm_before_head:
            assert stages_after_norm, "[R2n] stages_norm_before_head only applies to R2 (stages_after_norm)"
        # [NEW - R3] the stages go in exactly one of three places: before middle (the default),
        #   after_norm, or after_head. With two of them on, the branch order in forward would
        #   silently take just one, so stop here instead.
        self.stages_after_head = stages_after_head
        assert not (da_adapter and stages_after_head), "[R3/da] these structural modes are mutually exclusive"
        assert not (stages_after_norm and stages_after_head), \
            "[R3] stages_after_norm and stages_after_head are mutually exclusive - pick one position"
        # [NEW - R3a] a fourth position. The stages have to match the width of conv1's output
        #   (96->96, 1x1). With expand_encoder_head=True, enc_out_dim == z_dim*2 == that output,
        #   so they build at the same width as R3; an assert in GeopriorVAE holds that premise.
        #   They are not run here - encode() runs them.
        self.stages_after_conv1 = stages_after_conv1
        assert sum(bool(f) for f in (stages_after_norm, stages_after_head, stages_after_conv1)) <= 1, \
            "[R3a] only one stage-position flag may be set"
        assert not (da_adapter and stages_after_conv1), "[R3a/da] these structural modes are mutually exclusive"
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

        # [NEW] Added downsample stages between downsamples and middle.
        # [NEW - R3] stage width. Placed after the head conv, the input is not 384 (out_dim) but
        #   the head's output width, z_dim - the enc_out_dim the caller passes, e.g. 48*2 = 96.
        #   Note that pretrained_copy copies a 384-wide pretrained block, so it cannot work at a
        #   different width.
        _stage_dim = z_dim if (stages_after_head or stages_after_conv1) else out_dim
        assert not ((stages_after_head or stages_after_conv1) and any(
            (s.get('init') == 'pretrained_copy') for s in (add_stages or []))), \
            "[R3/R3a] init='pretrained_copy' does not work in this position: the pretrained block is 384 wide, the stage is not"
        self.add_downsamples = nn.ModuleList()
        if add_stages:
            for stage_cfg in add_stages:
                layers = []
                for _ in range(stage_cfg.get('num_res_blocks', 2)):
                    layers.append(ResidualBlock(_stage_dim, _stage_dim, dropout))
                resample = Resample(_stage_dim, mode=stage_cfg['mode'])
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
        # [NEW - da_adapter] the head conv is the channel compressor: 384 -> z_dim/r^2, at full
        #   resolution. The spatial fold is pixel_unshuffle(r) in forward, giving z_dim channels
        #   on a grid divided by r, so the latent shape is unchanged and mid stays at full
        #   resolution. b_adaptive attaches to this conv as usual, keeping the head[-1] slot so
        #   the trainer's reference to head still resolves.
        _head_out = z_dim
        if da_adapter:
            assert not add_stages, "[da_adapter] cannot be combined with add_encoder_stages - it replaces the stages"
            assert z_dim % (da_spatial_fold ** 2) == 0, f"z_dim {z_dim} % r²={da_spatial_fold**2} != 0"
            _head_out = z_dim // (da_spatial_fold ** 2)
        # [NEW/B-fix] with use_b_adaptive=True the last conv becomes AdaptiveWeightedCausalConv3d, the (B) weight gradient ratio mechanism
        if use_b_adaptive:
            # local import, to avoid a circular import
            from adaptive_weighted_causal_conv_3d import AdaptiveWeightedCausalConv3d
            _last_conv = AdaptiveWeightedCausalConv3d(
                out_dim, _head_out, 3, padding=1,
                adaptive_weight_eps=b_adaptive_eps,
                adaptive_weight_max=b_adaptive_max,
                disc_weight=b_adaptive_disc_weight,
            )
        else:
            _last_conv = CausalConv3d(out_dim, _head_out, 3, padding=1)
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            _last_conv)
        # [NEW - R2n] the re-normalisation module. Off means it is never built, so the state_dict is unchanged.
        if self.stages_norm_before_head:
            self.stages_renorm = RMS_norm(out_dim, images=False)

    def _run_add_stages(self, x, feat_cache=None, feat_idx=[0], return_skip=False, ff_skips=None):
        # [NEW - R2] pull the add_downsamples blocks out, so stages_after_norm can choose where they run.
        #   the cache-aware code and the Option-B chunk-equivalence skip stay as they are; routing
        #   this through the isinstance branch in the head loop would lose the cache
        ## [NEW/skip] added downsample stages with skip connection
        for stage in self.add_downsamples:
            x_in = x
            B, C, T_in, H, W = x_in.shape
            for layer in stage:
                # [crossattn] the feature just before add_downsamples' Resample(downsample) is skip@32
                if return_skip and isinstance(layer, Resample) and 'downsample' in layer.mode:
                    ff_skips.append(x)
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

            resample_mode = next((l.mode for l in stage if isinstance(l, Resample)), 'none')
            if resample_mode == 'downsample3d':
                # Option-B: chunked-aware skip that produces identical output for
                # chunked (T_in pattern 1,2,2,...) and single-pass (T_in odd >1).
                # - T_in == 1  : spatial-only mean-4 (chunked chunk 0)
                # - T_in even  : pixel_unshuffle_3d + mean-8 (chunked chunks 1+, matches working repo)
                # - T_in odd>1 : first frame mean-4 + rest pair-wise mean-8 (single-pass replica)
                if T_in == 1:
                    skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                    skip = F.pixel_unshuffle(skip, 2)
                    skip = skip.view(B, C, 4, H // 2, W // 2).mean(dim=2)
                    skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
                elif T_in % 2 == 0:
                    skip = pixel_unshuffle_3d(x_in, 2)
                    skip = skip.view(B, C, 8, T_in // 2, H // 2, W // 2).mean(dim=2)
                else:
                    skip0 = rearrange(x_in[:, :, :1], 'b c t h w -> (b t) c h w')
                    skip0 = F.pixel_unshuffle(skip0, 2)
                    skip0 = skip0.view(B, C, 4, H // 2, W // 2).mean(dim=2).unsqueeze(2)
                    rest = x_in[:, :, 1:]
                    T_rest = rest.shape[2]
                    skip_rest = pixel_unshuffle_3d(rest, 2)
                    skip_rest = skip_rest.view(B, C, 8, T_rest // 2, H // 2, W // 2).mean(dim=2)
                    skip = torch.cat([skip0, skip_rest], dim=2)
            elif resample_mode == 'downsample2d':
                skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                skip = F.pixel_unshuffle(skip, 2)
                skip = skip.view(B * T_in, C, 4, H // 2, W // 2).mean(dim=2)
                skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
            elif resample_mode == 'downsample_temporal':
                T_out = x.shape[2]
                if feat_cache is not None:
                    # chunked: per-chunk first T_out (origin behaviour)
                    skip = x_in[:, :, :T_out, :, :]
                else:
                    # [single-pass fix] strided index = [0] + [2k-1] to match chunked total frame mapping
                    t_idx = [0] + [2 * k - 1 for k in range(1, T_out)]
                    skip = x_in[:, :, t_idx, :, :]
            else:
                skip = x_in
            x = x + skip

        return x

    def forward(self, x, feat_cache=None, feat_idx=[0], return_skip=False):
        # [crossattn] with return_skip=True, collect the feature from *before* each downsample and
        #   return (the usual output, [skip@256, skip@128, skip@64, skip@32]). With False the
        #   behaviour is identical to before. Each skip is x just ahead of a Resample(downsample),
        #   the highest-quality feature at that resolution.
        ff_skips = [] if return_skip else None
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
            # [NEW 2026-07-21 gc-v2] checkpoint the first full-resolution conv activation too, under the same condition as Resample and Residual
            if _EXTENDED_VAE_GC and torch.is_grad_enabled():
                x = checkpoint(self.conv1, x, use_reentrant=False)
            else:
                x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            # [crossattn] the feature just before a downsample Resample is that resolution's skip
            if return_skip and isinstance(layer, Resample) and 'downsample' in layer.mode:
                ff_skips.append(x)
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## [NEW/skip] added downsample stages with skip connection
        # [NEW - R2] with stages_after_norm=True, skip here and run after middle -> norm -> SiLU.
        # [NEW - R3] stages_after_head=True skips here as well. Without this guard the stages run
        #   *a second time* here under R3, feeding a 384-wide input into a 96-wide block and
        #   failing with RuntimeError "size of tensor a (384) must match b (96)".
        # [NEW - R3a] stages_after_conv1 skips too - under R3a it is encode() that runs the stages.
        #   This is the exact spot where a missing guard made them run twice under R3.
        if not self.stages_after_norm and not self.stages_after_head and not self.stages_after_conv1:
            x = self._run_add_stages(x, feat_cache, feat_idx, return_skip, ff_skips)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        # [NEW/B-fix] with use_b_adaptive=True the last layer is AdaptiveWeightedCausalConv3d, whose
        #   forward can return a (y_main, y_adv) tuple. The caller, GeopriorVAE.encode, handles it.
        # [NEW - R2] stages_after_norm runs the stages after norm and SiLU, where the features are
        #   native to the pretrained model, leaving only the head conv behind them
        if self.stages_after_norm:
            x = self.head[0](x)
            x = self.head[1](x)
            x = self._run_add_stages(x, feat_cache, feat_idx, return_skip, ff_skips)
            if self.stages_norm_before_head:
                x = self.stages_renorm(x)   # [NEW - R2n] re-normalise the stack output, restoring the head conv's input anchor
            _head_layers = [self.head[2]]
        else:
            _head_layers = self.head
        _da_pre = None
        for layer in _head_layers:
            # [NEW - da_adapter] capture the feature just before the compressing conv; the shortcut
            #   sees the same input the conv does, as in DA-VAE's DADownBlock
            if self.da_adapter and layer is self.head[-1]:
                _da_pre = x
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
        # [NEW/B-fix] at inference (feat_cache is not None) there is no backward, so take main only.
        # In training (feat_cache is None) with use_b_adaptive=True, pass the tuple on to the caller.
        if isinstance(x, tuple) and feat_cache is not None:
            x = x[0]
        # [NEW - R3] stages_after_head runs the stages only *after* the head conv.
        #   The point is that the head conv's input goes back to being SiLU(RMS_norm(.)) and so is
        #   anchored again. R2 fed it the stage output directly, with no normalisation, and that is
        #   where grad_W and grad_y jumped 15x in the 2026-08-13 divergence.
        #   Two side effects: the head conv now runs at the pretrained resolution (16^2), and the
        #   stage width drops from 384 to z_dim.
        #   Note that under b_adaptive training x is a (y_main, y_adv) tuple, and both branches have
        #   to go through the same stages - the same reason da_adapter does this just below. The
        #   stage weights then receive gradients from both paths; under R2 this sat before the
        #   split and received them once.
        if self.stages_after_head:
            if isinstance(x, tuple):
                x = tuple(self._run_add_stages(_x, feat_cache, feat_idx, False, None) for _x in x)
            else:
                x = self._run_add_stages(x, feat_cache, feat_idx, return_skip, ff_skips)
        # [NEW - da_adapter] spatial fold plus a weight-free shortcut: unshuffle (384 -> 384r^2),
        #   then average into z_dim groups. Under b_adaptive training this is a (y_main, y_adv)
        #   tuple, and both outputs take the same fold, sharing the shortcut.
        if self.da_adapter:
            _r = self.da_fold
            _skip = _spatial_unshuffle3d(_da_pre, _r)
            _g = _skip.shape[1] // self.z_dim
            _skip = _skip.view(_skip.shape[0], self.z_dim, _g, *_skip.shape[2:]).mean(dim=2)
            if isinstance(x, tuple):
                x = tuple(_spatial_unshuffle3d(_x, _r) + _skip for _x in x)
            else:
                x = _spatial_unshuffle3d(x, _r) + _skip
        # [crossattn] with return_skip=True, return the skip list alongside the usual x
        if return_skip:
            return x, ff_skips
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
                 add_tail_stages=None,  # [NEW/geoprior] stages after head (3ch RGB space)
                 add_before_head_stages=None,  # [NEW/geoprior] upsample stages just before head (feature space)
                 da_adapter=False,      # [NEW - da_adapter] DA-VAE da_up-style entry: conv -> pixel_shuffle plus a repeat shortcut, with mid at full resolution
                 da_base_split=False,   # [NEW - da_adapter] split off the base (z_prior) share: keep the pretrained conv1 (16->384) and join with a nearest x r upsample
                 da_spatial_fold=4,     # [NEW - da_adapter] spatial expansion factor r
                 upsample_stages_before_middle=False,  # [NEW - R2/M1] run add_upsamples before middle (the mirror of R2)
                 upsample_stages_before_conv1=False,   # [NEW - R3a mirror] run add_upsamples in the z space (64ch) *before* conv1, so the pretrained conv1 runs at full resolution
                 dual_branch=False,   # [NEW/geoprior]
                 prior_z_dim=None,    # [NEW/geoprior] None → same as z_dim (symmetric)
                 expand_conv2=True,   # [NEW/geoprior] True: conv2 z_dim→z_dim → decoder input z_dim+prior_z_dim
                 first_frame_inject=False,  # [crossattn] inject the first frame as a keyframe through cross-attention
                 ff_inject_levels="all",    # which upsample levels to inject into ("all", or e.g. "2,3")
                 ff_window=32,              # windowed cross-attn window size
                 ff_encoder_source='residual',  # [crossattn base] which encoder supplies the cross-attn features: residual (self.encoder, 4 levels) or base (prior_encoder, 3 levels, no @32)
                 ff_single_level=0,         # [crossattn single-level ablation] 0 keeps the per-level symmetry (the default); 256/128/64/32 injects that one encoder skip into every decoder level
                 ff_dual_source=False):     # [dual 2026-07-14] mount both sources at once: residual at 4 levels and base (prior_encoder) at 3 (L1-L3), as two separate sets of gated blocks
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

        # [NEW/geoprior] dual_branch decoder input channels:
        #   expand_conv2=True:  conv2(z_dim→z_dim) + conv2_prior(prior_z_dim→prior_z_dim) → z_dim+prior_z_dim
        #   expand_conv2=False: conv2(z_dim→prior_z_dim) + conv2_prior(prior_z_dim→prior_z_dim) → prior_z_dim*2
        if dual_branch:
            _prior_z = prior_z_dim if prior_z_dim is not None else z_dim
            in_z = (z_dim + _prior_z) if expand_conv2 else (_prior_z * 2)
        else:
            _prior_z = None
            in_z = z_dim
        # [NEW - da_adapter] DA-VAE da_up-style entry: replace conv1 (in_z -> 384) at /r with a conv to 384r^2 followed by pixel_shuffle.
        #   With da_base_split=True the base (z_prior) share keeps its own conv1 (16->384), the same
        #   shape as the pretrained one, so loading the state_dict transplants the pretrained
        #   dec.conv1 automatically; the two then join at 32^2 through a nearest x r upsample.
        self.da_adapter = da_adapter
        self.da_base_split = da_base_split and dual_branch
        self.da_fold = da_spatial_fold
        # [NEW - R2/M1]
        self.upsample_stages_before_middle = upsample_stages_before_middle
        assert not (da_adapter and upsample_stages_before_middle), "[R2/da] these structural modes are mutually exclusive"
        assert not (upsample_stages_before_middle and add_before_head_stages), \
            "[R2/M1] the mirror cannot be combined with before_head stages - drop them, the mirror replaces them"
        if da_adapter:
            assert not add_stages and not add_before_head_stages, \
                "[da_adapter] cannot be combined with add_decoder(_before_head)_stages - it replaces the entry and mid placement"
            _da_in = (in_z - _prior_z) if self.da_base_split else in_z
            _da_out = dims[0] * da_spatial_fold ** 2
            assert _da_out % _da_in == 0, f"the repeat shortcut needs a whole multiple: {_da_out} % {_da_in} != 0"
            self.da_up = CausalConv3d(_da_in, _da_out, 3, padding=1)
            self._da_in = _da_in
            self._da_repeats = _da_out // _da_in
            if self.da_base_split:
                self.conv1 = CausalConv3d(_prior_z, dims[0], 3, padding=1)
            else:
                self.conv1 = None
        else:
            self.conv1 = CausalConv3d(in_z, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # [NEW] Added upsample stages between middle and upsamples.
        # [NEW - R3a mirror] placed before conv1, the width is conv1's input (in_z = 64, z_main 48
        #   plus z_prior 16). This is the decoder counterpart of the encoder's R3/R3a, where putting
        #   z-width stages after the pretrained conv measured +1.1 to +1.4 dB.
        #   Only one position flag, no da_adapter or pretrained_copy, and keepdim mode only: the
        #   widening variant doubles the channels and no longer matches conv1.
        self.upsample_stages_before_conv1 = upsample_stages_before_conv1
        if upsample_stages_before_conv1:
            assert not upsample_stages_before_middle, "[R3a mirror] only one decoder stage position may be set"
            assert not da_adapter, "[R3a mirror/da] these are mutually exclusive"
            assert add_stages, "[R3a mirror] needs add_decoder_stages - the stages are the upsampler"
            assert all(s.get('mode') == 'upsample2d_keepdim' for s in add_stages), \
                "[R3a mirror] only upsample2d_keepdim is supported - the widening variant does not match conv1's input width"
            assert not any(s.get('init') == 'pretrained_copy' for s in add_stages), \
                "[R3a mirror] pretrained_copy does not work here: the pretrained block is 384 wide, this is 64"
        inner_dim = in_z if upsample_stages_before_conv1 else dims[0]
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
                # [NEW - R2/M1] width-preserving mirror: both ResBlock and Resample stay dim -> dim
                # (the else branch below is the variant that doubles the channels)
                elif mode == 'upsample2d_keepdim':
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

        # [crossattn 2026-07] first-frame keyframe cross-attention injection.
        #   The context now comes from the frozen VAE encoder's return_skip intermediates; the
        #   hand-written pyramid was dropped.
        #   One injection point per upsample resolution level, after that level's last
        #   ResidualBlock, giving a symmetric four-level arrangement. Level boundaries are found
        #   by looking for Resample, so each point sits just before the next one.
        #   query_dim is the decoder feature width there, dims[level+1], measured as
        #   [384@32, 384@64, 192@128, 96@256]; context_dim is the encoder skip width at the
        #   matching resolution, the same [384@32, 384@64, 192@128, 96@256].
        #   The encoder skips are collected fine to coarse:
        #   [skip@256 (96), skip@128 (192), skip@64 (384), skip@32 (384)].
        #   → decoder level 0(coarse,@32) = skip idx 3, level 1(@64)=idx 2, level 2(@128)=idx 1, level 3(@256)=idx 0.
        self.first_frame_inject = first_frame_inject
        self.ff_encoder_source = ff_encoder_source  # [NEW base] residual | base
        self.ff_single_level = int(ff_single_level)  # [NEW single-level] 0 is off; 256/128/64/32 injects that one skip into every level
        # [NEW single-level ablation] the encoder skips are collected fine to coarse: idx0 = @256
        #   (96ch), idx1 = @128 (192ch), idx2 = @64 (384ch), idx3 = @32 (384ch). This maps a
        #   resolution to (skip_idx, channels).
        _single_map = {256: (0, 96), 128: (1, 192), 64: (2, 384), 32: (3, 384)}
        self.ff_level_channels = dims[1:]         # [384,384,192,96] — decoder level(coarse→fine) out_ch
        self.ff_inject = nn.ModuleDict()          # key=str(upsamples flat idx), val=GatedCrossAttnBlock
        self.ff_inject_skip = {}                  # flat idx -> index into the encoder skip list (fine to coarse)
        if first_frame_inject:
            from crossattn_ff import GatedCrossAttnBlock
            _nlv = len(dim_mult)                  # four levels
            if str(ff_inject_levels).lower() == "all":
                _lv = set(range(_nlv))
            else:
                _lv = set(int(s) for s in str(ff_inject_levels).split(",") if s.strip() != "")
            # find the flat index of each upsample level's last ResidualBlock: just before the next
            # Resample, or the very end for the last level.
            _last_resblock_of_level = {}          # level -> flat idx of its last ResidualBlock
            _flat, _level = 0, 0
            for _m in self.upsamples:
                if isinstance(_m, ResidualBlock):
                    _last_resblock_of_level[_level] = _flat
                elif isinstance(_m, Resample):
                    _level += 1
                _flat += 1
            for _level in range(_nlv):
                if _level not in _lv or _level not in _last_resblock_of_level:
                    continue
                # [NEW base] the base encoder (prior_encoder) has no add_downsample, so it has no
                #   deepest @32 skip. Injection into decoder coarse level 0 (@32) is therefore
                #   skipped, leaving three levels. The skip_idx mapping (levels 1,2,3 -> 2,1,0)
                #   lines up with the base skip list [256,128,64] at idx 0,1,2: @64->2, @128->1,
                #   @256->0.
                if ff_encoder_source == 'base' and _level == 0:
                    continue
                _flat = _last_resblock_of_level[_level]
                _out = dims[_level + 1]           # query dim = the decoder feature width at this level, kept per level
                if self.ff_single_level and self.ff_single_level in _single_map:
                    # [NEW single-level ablation] inject one chosen encoder skip into every decoder
                    #   level. The query dim (_out) stays per level; only the context is unified to
                    #   that single skip, at a fixed index and width. CrossAttention's window
                    #   scaling absorbs the resolution mismatch.
                    _skip_idx, _ctx = _single_map[self.ff_single_level]
                else:
                    # the default is symmetric per level. Since skips are collected fine to coarse,
                    # coarse level 0 maps to skip index nlv-1.
                    _skip_idx = _nlv - 1 - _level
                    # context dim = the encoder skip width at the matching resolution; measured
                    # L0=384, L1=384, L2=192, L3=96.
                    _ctx = dims[_level + 1]
                self.ff_inject[str(_flat)] = GatedCrossAttnBlock(
                    dim=_out, context_dim=_ctx, window_size=(ff_window, ff_window))
                self.ff_inject_skip[_flat] = _skip_idx

        # [dual 2026-07-14] dual-source: mount the residual blocks above and the base
        #   (prior_encoder) blocks at the same time. The base encoder has no add_downsample, so it
        #   has three skips (fine to coarse [@256, @128, @64]) and covers L1-L3, not L0.
        #   The mapping and widths match the existing ff_encoder_source='base' path:
        #   level -> skip idx = nlv-1-level, context = dims[level+1].
        #   gamma starts at 0, so mounting them does not change the output. In forward they are
        #   summed in sequence, after the residual injection.
        self.ff_dual_source = bool(ff_dual_source)
        self.ff_inject_base = nn.ModuleDict()
        self.ff_inject_base_skip = {}
        if first_frame_inject and self.ff_dual_source:
            from crossattn_ff import GatedCrossAttnBlock as _GCB
            _nlv = len(dim_mult)
            _last_rb = {}
            _flat, _level = 0, 0
            for _m in self.upsamples:
                if isinstance(_m, ResidualBlock):
                    _last_rb[_level] = _flat
                elif isinstance(_m, Resample):
                    _level += 1
                _flat += 1
            for _level in range(1, _nlv):            # base has no @32 skip, so L0 is excluded
                if _level not in _last_rb:
                    continue
                _flat = _last_rb[_level]
                _out = dims[_level + 1]
                self.ff_inject_base[str(_flat)] = _GCB(
                    dim=_out, context_dim=dims[_level + 1], window_size=(ff_window, ff_window))
                self.ff_inject_base_skip[_flat] = _nlv - 1 - _level

        # [NEW/geoprior] add_before_head: DC-AE style upsample just before head (out_dim feature space)
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

        # [NEW] Deferred pretrained_copy init for add_upsamples
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

        # [NEW/geoprior] add_tail: stages after head in 3ch RGB space
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

    def _run_add_upsamples(self, x, feat_cache=None, feat_idx=[0]):
        # [NEW - R2/M1] pull the add_upsamples blocks out, so upsample_stages_before_middle can choose where they run
        ## [NEW/skip] added upsample stages with skip connection
        for stage in self.add_upsamples:
            x_in = x
            B, C, T_in, H, W = x_in.shape
            for layer in stage:
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)

            resample_mode = next((l.mode for l in stage if isinstance(l, Resample)), 'none')
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
            elif resample_mode in ('upsample2d', 'upsample2d_keepdim'):
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

        return x

    def _cached_conv(self, conv, x, feat_cache, feat_idx):
        # [NEW - da_adapter] shared helper for applying the CausalConv3d cache - the same logic the conv1 block uses
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
            y = conv(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
            return y
        return conv(x)

    def forward(self, x, feat_cache=None, feat_idx=[0], ff_skips=None, ff_skips_base=None):
        ## conv1  ([NEW - da_adapter] the da_up entry branch)
        if self.da_adapter:
            # concatenation order is [z_main(0:z_dim), z_prior(z_dim:)], as in WanVAE_.decode
            if self.da_base_split:
                xm, xb = x[:, :self._da_in], x[:, self._da_in:]
            else:
                xm, xb = x, None
            h = self._cached_conv(self.da_up, xm, feat_cache, feat_idx)
            h = _spatial_shuffle3d(h, self.da_fold)
            # weight-free shortcut (DA-VAE's DAUpBlock): repeat the channels, then shuffle
            _sc = xm.repeat_interleave(self._da_repeats, dim=1)
            h = h + _spatial_shuffle3d(_sc, self.da_fold)
            if xb is not None:
                # base path: the pretrained conv1 (16->384) at /r, then a nearest x r upsample to join at full resolution
                b = self._cached_conv(self.conv1, xb, feat_cache, feat_idx)
                _B = b.shape[0]
                b = rearrange(b, 'b c t h w -> (b t) c h w')
                b = F.interpolate(b, scale_factor=self.da_fold, mode='nearest')
                b = rearrange(b, '(b t) c h w -> b c t h w', b=_B)
                h = h + b
            x = h
        else:
            # [NEW - R3a mirror] run the stages in the z space (64ch) *before* conv1, so conv1 runs
            #   at full resolution. With zero-init, step 0 leaves only the repeat + pixel_shuffle
            #   skip, a weight-free nearest-style upsample.
            if self.upsample_stages_before_conv1:
                x = self._run_add_upsamples(x, feat_cache, feat_idx)
            x = self._cached_conv(self.conv1, x, feat_cache, feat_idx)

        # [NEW - R2/M1] the mirror: run the upsample stages before middle, so mid runs at full resolution (32^2)
        if self.upsample_stages_before_middle:
            x = self._run_add_upsamples(x, feat_cache, feat_idx)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## [NEW/skip] added upsample stages with skip connection
        # [NEW - R3a mirror] with before_conv1 the stages have already run, so skip them here too
        #   (the legacy after-middle position). This is the same spot where a missing guard caused
        #   the double-run shape error in the encoder's R3.
        if not self.upsample_stages_before_middle and not self.upsample_stages_before_conv1:
            x = self._run_add_upsamples(x, feat_cache, feat_idx)

        ## upsamples  (+ [crossattn] keyframe cross-attention injected after each ResidualBlock)
        for _u, layer in enumerate(self.upsamples):
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
            # the gate (gamma) starts at 0, so the first forward is bit-identical even with
            # ff_skips present. No feat_cache here, following the AttentionBlock pattern.
            #   ff_skips[i] is an encoder skip (2D, B,C,H',W') at the matching resolution, used as
            #   the cross-attention context.
            if self.first_frame_inject and ff_skips is not None and str(_u) in self.ff_inject:
                x = self.ff_inject[str(_u)](x, ff_skips[self.ff_inject_skip[_u]])
            # [dual] the base (prior_encoder) source blocks, summed in sequence after the residual
            #   injection: x = x + gamma_res * attn_res(...), then x = x + gamma_base * attn_base(...).
            #   Both gammas start at 0, so mounting them has no effect.
            if self.first_frame_inject and ff_skips_base is not None and str(_u) in self.ff_inject_base:
                x = self.ff_inject_base[str(_u)](x, ff_skips_base[self.ff_inject_base_skip[_u]])

        ## [NEW/geoprior] add_before_head: DC-AE style upsample in feature space (before head)
        for stage in self.add_before_head:
            x_in = x
            B, C, T_in, H, W = x_in.shape
            for layer in stage:
                if feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)
            resample_mode = next((l.mode for l in stage if isinstance(l, Resample)), 'none')
            if resample_mode == 'upsample3d':
                if feat_cache is not None:
                    # [FIX 2026-06-27, the root cause of the dropped frames] in a chunked decode the
                    #   main Resample produces a full 2x temporal (2t) per chunk. Without this chunk
                    #   branch the single-pass formula, 1 + (t-1)*2 = 2t-1, was applied to chunked
                    #   decoding too, leaving the skip one frame short of main's 2t and losing frames
                    #   at x + skip (17 -> 15, 81 -> 71). Adding the chunk branch, as the
                    #   add_upsamples skip at lines 744-752 already does, makes the lengths agree.
                    if T_in == 1:
                        # first chunk: Resample 'Rep' keeps the time axis, so only space doubles
                        skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                        skip = skip.repeat_interleave(4, dim=1)
                        skip = F.pixel_shuffle(skip, 2)
                        skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
                    else:
                        # later chunks: a full 2x on time (2t), matching main's length
                        skip = x_in.repeat_interleave(8, dim=1)
                        skip = pixel_shuffle_3d(skip, 2)
                else:
                    # 2D spatial pixel_shuffle per frame. For T_in>1 (single-pass), expand
                    # temporally to match T_out: first frame as-is, subsequent frames each
                    # repeated 2x (matching chunked per-chunk broadcast behaviour).
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
                    # [FIX] chunked: a full 2x on time, the same as main
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
        # [NEW 2026-07-21 gc-v2] checkpoint the full-resolution head as well, when feat_cache is
        #   None and gradients are needed. Do not early-return here: the add_tail step after the
        #   head has to stay reachable, and with checkpointing the loop below simply runs empty.
        _head_gc = _EXTENDED_VAE_GC and feat_cache is None and torch.is_grad_enabled()
        if _head_gc:
            def _head_fn(_x):
                for _l in self.head:
                    _x = _l(_x)
                return _x
            x = checkpoint(_head_fn, x, use_reentrant=False)
        for layer in (self.head if not _head_gc else []):
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

        ## [NEW/geoprior] add_tail: DC-AE style refinement in 3ch RGB space
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


# [NEW - da_adapter] spatial-only shuffle and unshuffle: the time axis is untouched, and the
# channel layout is channel-major, out_ch = c*r^2 + phase.
def _spatial_unshuffle3d(x, r):
    b = x.shape[0]
    y = rearrange(x, 'b c t h w -> (b t) c h w')
    y = F.pixel_unshuffle(y, r)
    return rearrange(y, '(b t) c h w -> b c t h w', b=b)


def _spatial_shuffle3d(x, r):
    b = x.shape[0]
    y = rearrange(x, 'b c t h w -> (b t) c h w')
    y = F.pixel_shuffle(y, r)
    return rearrange(y, '(b t) c h w -> b c t h w', b=b)


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
                 add_decoder_tail_stages=None,          # [NEW/geoprior] stages after head (3ch RGB space)
                 add_decoder_before_head_stages=None,   # [NEW/geoprior] upsample stages just before head
                 dual_branch=False,           # [NEW/geoprior]
                 subsample_mode='avg_pool',   # [NEW/geoprior] 'avg_pool' | 'stride' | 'bilinear'
                 prior_z_dim=None,            # [NEW/geoprior] None → same as z_dim; int for asymmetric (e.g. 16 for frozen Wan)
                 expand_conv2=True,           # [NEW/geoprior] True: conv2 z_dim→z_dim; False: conv2 z_dim→prior_z_dim (old)
                 expand_encoder_head=False,   # [NEW/geoprior] True: encoder.head outputs z_dim*2 (instead of prior_z_dim*2)
                 use_b_adaptive=False,        # [NEW/B-fix] encoder.head[-1] = AdaptiveWeightedCausalConv3d (= single backward weight ratio)
                 b_adaptive_eps=1e-6,
                 b_adaptive_max=1e7,
                 b_adaptive_disc_weight=1.0,
                 da_adapter=False,            # [NEW - da_adapter] DA-VAE-style boundary adapter: channel compression plus unshuffle in the encoder, da_up in the decoder
                 da_base_split=False,         # [NEW - da_adapter] keep the decoder's base share separate (variant b')
                 da_spatial_fold=4,           # [NEW - da_adapter] the spatial fold and expansion factor r
                 stages_after_norm=False,     # [NEW - R2] encoder stages after norm and SiLU, plus the decoder M1 mirror - one flag for the pair
                 stages_norm_before_head=False,  # [NEW - R2n]
                 stages_after_head=False,     # [NEW - R3] encoder stages after the head conv; the decoder keeps R2's placement
                 stages_after_conv1=False,    # [NEW - R3a] stages after vae.conv1, just before chunking, so the pretrained head -> conv1 pair stays together at full resolution
                 dec_stages_before_conv1=False,  # [NEW - R3a mirror] decoder stages in the z space (64ch) before conv1, so the pretrained dec.conv1 runs at full resolution
                 decoder_mirror=True,         # [NEW - R2 control] False moves the encoder only
                 first_frame_inject=False,    # [crossattn] inject the first frame as a keyframe through cross-attention
                 ff_inject_levels="all",      # which upsample levels to inject into
                 ff_window=32,                # windowed cross-attn window size
                 ff_encoder_source='residual',  # [crossattn base] residual(self.encoder) | base(prior_encoder full-res)
                 ff_single_level=0,           # [crossattn single-level ablation] 0 is off; 256/128/64/32 injects that one skip into every level
                 ff_dual_source=False):       # [dual 2026-07-14] mount the residual and base references at once
        super().__init__()
        self.first_frame_inject = first_frame_inject
        self.ff_encoder_source = ff_encoder_source  # [NEW base]
        self.ff_dual_source = bool(ff_dual_source)  # [NEW dual]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.dual_branch = dual_branch
        self.subsample_mode = subsample_mode
        # [NEW/geoprior] prior branch z_dim (Wan fixed = 16); None → symmetric
        self.prior_z_dim = prior_z_dim if prior_z_dim is not None else z_dim

        self.expand_encoder_head = expand_encoder_head
        # modules
        # [NEW/geoprior] dual_branch: encoder head fixed at prior_z_dim*2 (same as pretrained Wan)
        # → encoder.head[-1] keeps shape (384→prior_z_dim*2), no weight mismatch
        # expand_encoder_head=True: encoder.head outputs z_dim*2 (true channel expansion through head)
        enc_out_dim = (self.prior_z_dim * 2) if (dual_branch and not expand_encoder_head) else (z_dim * 2)
        # [NEW - da_adapter] keep the flags and pass them to the main encoder and decoder only; prior_encoder stays stock
        self.da_adapter = da_adapter
        self.da_base_split = da_base_split
        self.da_spatial_fold = da_spatial_fold
        self.stages_after_norm = stages_after_norm  # [NEW - R2]
        self.stages_after_head = stages_after_head  # [NEW - R3]
        self.stages_after_conv1 = stages_after_conv1  # [NEW - R3a]
        # [NEW - R3a] Encoder3d builds the stages at its own z_dim (enc_out_dim), while conv1's
        #   output is z_dim*2. For the two to match, expand_encoder_head is required - that is when
        #   enc_out_dim == z_dim*2. Otherwise conv1's output (96) and the stage width (32) disagree
        #   and the first forward dies, so fail here instead.
        if stages_after_conv1:
            assert expand_encoder_head, \
                "[R3a] stages_after_conv1 requires expand_encoder_head=True, so the stage width equals conv1's output width"
        self.encoder = Encoder3d(dim, enc_out_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout,
                                 add_stages=add_encoder_stages,
                                 stages_after_head=stages_after_head,  # [NEW - R3]
                                 stages_after_conv1=stages_after_conv1,  # [NEW - R3a]
                                 da_adapter=da_adapter,
                                 da_spatial_fold=da_spatial_fold,
                                 stages_after_norm=stages_after_norm,  # [NEW - R2]
                                 stages_norm_before_head=stages_norm_before_head,  # [NEW - R2n]
                                 use_b_adaptive=use_b_adaptive,
                                 b_adaptive_eps=b_adaptive_eps,
                                 b_adaptive_max=b_adaptive_max,
                                 b_adaptive_disc_weight=b_adaptive_disc_weight)
        # conv1: asymmetric (prior_z_dim*2 → z_dim*2) when dual_branch, square otherwise
        self.conv1 = CausalConv3d(enc_out_dim, z_dim * 2, 1)
        # conv2: expand_conv2=True → z_dim→z_dim; False → z_dim→prior_z_dim (old behavior)
        # decoder.conv1 input: expand_conv2=True → z_dim+prior_z_dim; False → prior_z_dim*2
        self.expand_conv2 = expand_conv2
        if dual_branch:
            _conv2_out = z_dim if expand_conv2 else self.prior_z_dim
            self.conv2 = CausalConv3d(z_dim, _conv2_out, 1)
            if z_dim != self.prior_z_dim:
                # [NEW/geoprior] separate frozen conv2 for prior branch
                self.conv2_prior = CausalConv3d(self.prior_z_dim, self.prior_z_dim, 1)
        else:
            self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout,
                                 add_stages=add_decoder_stages,
                                 add_tail_stages=add_decoder_tail_stages,
                                 add_before_head_stages=add_decoder_before_head_stages,  # [NEW/geoprior]
                                 da_adapter=da_adapter,          # [NEW - da_adapter]
                                 da_base_split=da_base_split,
                                 da_spatial_fold=da_spatial_fold,
                                 # [NEW - R3] stages_after_head has to be included here. Without it,
                                 #   turning R3 on makes this False and the decoder *silently* moves
                                 #   to 'after middle', so both encoder and decoder differ from R2
                                 #   and the single-variable comparison breaks. R3 is meant to move
                                 #   the encoder only and keep the decoder as R2 has it.
                                 # [NEW - R3a mirror] when before_conv1 is on, before_middle must be
                                 #   off - only one position. Otherwise a Decoder3d assert fires,
                                 #   which is the guard against a silent double placement.
                                 upsample_stages_before_middle=((stages_after_norm or stages_after_head or stages_after_conv1) and decoder_mirror and not dec_stages_before_conv1),  # [NEW - part of the R2/M1 pair; R3 and R3a keep R2's decoder placement]
                                 upsample_stages_before_conv1=dec_stages_before_conv1,  # [NEW - R3a mirror]
                                 dual_branch=dual_branch,
                                 prior_z_dim=self.prior_z_dim,   # [NEW/geoprior]
                                 expand_conv2=expand_conv2,       # [NEW/geoprior]
                                 first_frame_inject=first_frame_inject,  # [crossattn]
                                 ff_inject_levels=ff_inject_levels,
                                 ff_window=ff_window,
                                 ff_encoder_source=ff_encoder_source,  # [crossattn base]
                                 ff_single_level=ff_single_level,  # [crossattn single-level]
                                 ff_dual_source=ff_dual_source)   # [dual]

        # [crossattn 2026-07] the first-frame keyframe feature comes from the frozen VAE encoder's
        #   return_skip intermediates, so there is no separate ff_encoder and no new parameters.
        #   decode() reuses self.encoder.

        # [NEW/geoprior] the lower branch: a vanilla Wan encoder, frozen and without add_stages
        # prior branch uses prior_z_dim (Wan's fixed z_dim=16); weights copied in _video_vae_geoprior
        if dual_branch:
            self.prior_encoder = Encoder3d(dim, self.prior_z_dim * 2, dim_mult, num_res_blocks,
                                           attn_scales, self.temperal_downsample, dropout,
                                           add_stages=None)
            # separate projection for prior branch (conv1 handles main encoder only)
            self.prior_conv1 = CausalConv3d(self.prior_z_dim * 2, self.prior_z_dim * 2, 1)

        # [NEW] cache count_conv3d results — avoids full module-tree traversal every step
        self._cached_dec_conv_num = count_conv3d(self.decoder)
        self._cached_enc_conv_num = count_conv3d(self.encoder)
        if dual_branch:
            self._cached_prior_conv_num = count_conv3d(self.prior_encoder)

    def forward(self, x):
        # [NEW/B-fix] encode can return a tuple, ((mu, log_var), (mu_adv, log_var_adv)).
        # GeopriorVAE.forward is the ordinary reconstruction path and uses the main branch only.
        # Callers such as GeopriorDiTAlignModel.forward handle the adv branch when use_b_adaptive is on.
        encode_result = self.encode(x, scale=None)
        if isinstance(encode_result[0], tuple):
            (mu, log_var), _ = encode_result    # ignore the adv branch on the ordinary forward path
        else:
            mu, log_var = encode_result
        z = self.reparameterize(mu, log_var)
        if self.dual_branch:
            with torch.no_grad():
                z_prior = self._encode_prior(x)
            z = torch.cat([z, z_prior], dim=1)  # (B, 2*z_dim, T', H', W')
        x_recon = self.decode(z, scale=None)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        if self.training:
            out = self.encoder(x)
        else:
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
        # [NEW - R3a] run the stages *after* conv1 and *before* chunking.
        #   - conv1 then runs at 32^2 (full resolution), and the pretrained head -> conv1 pair stays
        #     as it was trained. conv1 is 1x1 and so resolution-independent, which means the move is
        #     expected to be a reparameterisation; testing that prediction is the point of the run.
        #   - this point is *after* the chunks are joined (torch.cat), so the stages' causal conv
        #     sees the full T at once. No feat_cache is needed and chunked and single-pass are
        #     equivalent by construction, without R2/R3's chunk-equivalence logic.
        #   - with a b_adaptive tuple, both branches go through the same stages and share their
        #     weights, the same convention R3 uses.
        def _r3a(y):
            return self.encoder._run_add_stages(y, None, [0]) if self.stages_after_conv1 else y
        # [NEW/B-fix] out can be a (y_main, y_adv) tuple, when use_b_adaptive=True and training
        if isinstance(out, tuple):
            out_main, out_adv = out
            mu_main, log_var_main = _r3a(self.conv1(out_main)).chunk(2, dim=1)
            mu_adv,  log_var_adv  = _r3a(self.conv1(out_adv)).chunk(2, dim=1)
            if scale is not None:
                if isinstance(scale[0], torch.Tensor):
                    _sc0 = scale[0].view(1, self.z_dim, 1, 1, 1)
                    _sc1 = scale[1].view(1, self.z_dim, 1, 1, 1)
                else:
                    _sc0, _sc1 = scale[0], scale[1]
                mu_main = (mu_main - _sc0) * _sc1
                mu_adv  = (mu_adv  - _sc0) * _sc1
            self.clear_cache()
            return (mu_main, log_var_main), (mu_adv, log_var_adv)

        mu, log_var = _r3a(self.conv1(out)).chunk(2, dim=1)
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu, log_var

    # [NEW/geoprior] encode the lower branch: downsample T, H and W by 2 according to
    # subsample_mode, then run the frozen prior_encoder to get mu_prior.
    def _encode_prior(self, x):
        if self.subsample_mode == 'avg_pool':
            # a CausalVAE treats the first frame separately, so T has to be odd for the latent T to
            # line up. avg_pool3d produces floor(T/2), so an odd T gets its last frame repeated to
            # make it even.
            if x.shape[2] % 2 == 1:
                x_pad = torch.cat([x, x[:, :, -1:, :, :]], dim=2)
            else:
                x_pad = x
            x_sub = F.avg_pool3d(x_pad, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        elif self.subsample_mode == 'spatial_avg_temporal_stride':
            # [NEW] spatial 2x avg + temporal stride 2 (no temporal averaging)
            # temporal averaging is dropped so the I2V first frame keeps its values; only half the
            # GT frames are used. An odd T gets its last frame repeated to make it even, and the
            # output T matches the 'avg_pool' mode.
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
        # [NEW/f32t4] match the prior subsample to the main add_stage, leaving time uncompressed:
        #   add_encoder uses downsample2d, which is spatial-only, so the prior shrinks space only
        #   and keeps time. The main and prior latents then have the same size and concatenate.
        #   After the base (x8 / x4):
        #   bilinear_s2t1 → spatial ×16, temporal ×4 (f16t4) / bilinear_s4t1 → spatial ×32, temporal ×4 (f32t4).
        elif self.subsample_mode == 'bilinear_s2t1':
            B, C, T, H, W = x.shape
            x_flat = rearrange(x, 'b c t h w -> (b t) c h w')  # time left uncompressed (the ::2 is gone)
            x_flat = F.interpolate(x_flat, scale_factor=0.5, mode='bilinear', align_corners=False)  # spatial ×2
            x_sub = rearrange(x_flat, '(b t) c h w -> b c t h w', b=B)
        elif self.subsample_mode == 'bilinear_s4t1':
            B, C, T, H, W = x.shape
            x_flat = rearrange(x, 'b c t h w -> (b t) c h w')  # time left uncompressed
            x_flat = F.interpolate(x_flat, scale_factor=0.25, mode='bilinear', align_corners=False)  # spatial x4, in one step
            x_sub = rearrange(x_flat, '(b t) c h w -> b c t h w', b=B)
        # [NEW 2026-08-02 AA] two x0.5 steps make the 4x reduction anti-aliased. Doing s4t1 in one
        #   step measured twice the high-frequency aliasing, and swapping this in at eval time on an
        #   s4t1 checkpoint already trained gave +0.37 dB, improving 8 of 8 - so aliasing is a strong
        #   candidate for part of the f32 wall, the -2 dB against the scaling law.
        elif self.subsample_mode == 'bilinear_s4t1_2stage':
            B, C, T, H, W = x.shape
            x_flat = rearrange(x, 'b c t h w -> (b t) c h w')  # time left uncompressed
            x_flat = F.interpolate(x_flat, scale_factor=0.5, mode='bilinear', align_corners=False)
            x_flat = F.interpolate(x_flat, scale_factor=0.5, mode='bilinear', align_corners=False)
            x_sub = rearrange(x_flat, '(b t) c h w -> b c t h w', b=B)
        else:
            raise ValueError(f'Unknown subsample_mode: {self.subsample_mode}')

        t = x_sub.shape[2]
        # prior_encoder has no add_downsamples, so only the base transform applies
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

    def decode(self, z, scale, first_frame=None):
        self.clear_cache()
        if scale is not None:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        # [crossattn 2026-07] the first-frame keyframe becomes the frozen VAE encoder's intermediate
        #   features (return_skip). This is independent of chunking and encoded once. With
        #   first_frame None (T2V or ff_drop) or injection off it returns None and the original path
        #   runs. The encoder is a temporal CausalConv3d and needs T >= 3, so it encodes
        #   [first frame + zeros(T-1)]. Each skip is 5D (B,C,T',H,W), and slicing temporal index 0,
        #   the first frame, gives the 2D context.
        ff_skips = None
        ff_skips_base = None   # [dual]
        if first_frame is not None and getattr(self, 'first_frame_inject', False):
            # a single-pass encoder shrinks T through a time_conv (kernel 3, no padding) at every
            #   temporal downsample, and each step needs T >= 3 on its input. With n temporal
            #   downsamples a safe T is 4n+1. Measured: this geoprior config has 3, so T=9 passes,
            #   and temporal index 0 of every skip is always the first frame.
            _n_tdown = sum(1 for td in self.temperal_downsample if td)
            for stage in self.encoder.add_downsamples:
                for layer in stage:
                    if isinstance(layer, Resample) and layer.mode in ('downsample3d', 'downsample_temporal'):
                        _n_tdown += 1
            _T = max(3, 4 * _n_tdown + 1)
            _ff5 = torch.cat([
                first_frame[:, :, None],
                torch.zeros(first_frame.shape[0], first_frame.shape[1], _T - 1,
                            first_frame.shape[2], first_frame.shape[3],
                            device=first_frame.device, dtype=first_frame.dtype),
            ], dim=2)                                    # (B,3,_T,H,W)
            # [crossattn base] which feature encoder supplies the context:
            #   residual (the default): self.encoder, four skip levels including add_downsample
            #   base: self.prior_encoder, frozen pure Wan, with three skip levels (@256/128/64)
            #         because it has no add_downsample. The full-resolution _ff5 goes through as is,
            #         with no downsampling. There is no @32, and __init__ already excluded the
            #         decoder's @32 injection.
            if getattr(self, 'ff_encoder_source', 'residual') == 'base' and hasattr(self, 'prior_encoder'):
                _, _skips5 = self.prior_encoder(_ff5, return_skip=True)
            else:
                _, _skips5 = self.encoder(_ff5, return_skip=True)   # single-pass, no feat_cache
            # temporal index 0 of each skip is the first frame, giving 2D (B,C,H',W')
            ff_skips = [s[:, :, 0] for s in _skips5]
            # [dual] also collect the base skips, from prior_encoder at full resolution (frozen pure
            #   Wan). The call is the same as the 'base'-only path; here it runs alongside residual,
            #   and the L1-L3 blocks consume it.
            if getattr(self, 'ff_dual_source', False) and hasattr(self, 'prior_encoder'):
                _, _skips5b = self.prior_encoder(_ff5, return_skip=True)
                ff_skips_base = [s[:, :, 0] for s in _skips5b]
        # [NEW/geoprior] dual_branch: conv2 for z_main; conv2_prior (frozen) for z_prior
        if self.dual_branch:
            z_main = self.conv2(z[:, :self.z_dim])              # (B, prior_z_dim, T', H', W')
            if self.z_dim == self.prior_z_dim:
                z_p = self.conv2(z[:, self.z_dim:])             # shared conv2
            else:
                z_p = self.conv2_prior(z[:, self.z_dim:])       # separate frozen conv2_prior
            x = torch.cat([z_main, z_p], dim=1)                 # (B, prior_z_dim*2, T', H', W')
        else:
            x = self.conv2(z)
        # [NEW] force_single_pass decodes in one pass even in eval mode, instead of chunking, which
        #   avoids the 81-frame chunk bug. Chunking saves memory on long videos so it stays the
        #   default; turn this on only when a measurement has to be exact.
        # [NEW group-chunk] with decode_chunk_latent > 0, decode in chunks of K latent positions,
        #   in both training and eval. The point is to avoid the INT_MAX in the single-pass upsample
        #   backward when training at 256x81, without falling back to frame-by-frame, which is slow.
        #   feat_cache carries the context across boundaries, so this is equivalent to single-pass -
        #   verify it. The default of 0 keeps the existing behaviour.
        _chunk = getattr(self, 'decode_chunk_latent', 0)
        if _chunk and _chunk > 0:
            _outs = []
            for i in range(0, iter_, _chunk):
                self._conv_idx = [0]
                _outs.append(self.decoder(
                    x[:, :, i:i + _chunk, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx, ff_skips=ff_skips, ff_skips_base=ff_skips_base))
            out = torch.cat(_outs, 2)
        elif self.training or getattr(self, 'force_single_pass', False):
            out = self.decoder(x, ff_skips=ff_skips, ff_skips_base=ff_skips_base)
        else:
            for i in range(iter_):
                self._conv_idx = [0]
                if i == 0:
                    out = self.decoder(
                        x[:, :, i:i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx, ff_skips=ff_skips, ff_skips_base=ff_skips_base)
                else:
                    out_ = self.decoder(
                        x[:, :, i:i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx, ff_skips=ff_skips, ff_skips_base=ff_skips_base)
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


# [NEW/geoprior] dual-branch VAE loader
def _video_vae_geoprior(pretrained_path=None, z_dim=None, device='cpu',
                         add_encoder_stages=None, add_decoder_stages=None,
                         add_decoder_tail_stages=None,          # [NEW] stages after head (3ch RGB space)
                         add_decoder_before_head_stages=None,   # [NEW] upsample stages just before head
                         dual_branch=True, subsample_mode='avg_pool',
                         prior_z_dim=16,  # [NEW] fixed Wan z_dim for prior branch
                         decoder_conv1_zmain_init='zero',  # [NEW] 'zero' or 'pretrained' for z_main ch
                         zmain_fresh_init=False,   # [NEW - freshinit] drop the pretrained copy for the residual chain (head.2/conv1/conv2), leaving the default random init
                         fresh_gate_init='zero',   # [NEW - freshinit] the decoder.conv1 residual columns: 'zero' or 'random' (only meaningful with zmain_fresh_init=True)
                         zmain_hybrid_init=False,  # [NEW - hybrid] inherit the pretrained weights for z_main's first prior_z_dim channels only; the new channels stay random
                         da_adapter=False,         # [NEW - da_adapter] DA-VAE-style boundary adapter (variants b and b'); implies zmain_fresh_init
                         da_base_split=False,      # [NEW - da_adapter] variant b': keep the decoder's base share on its own pretrained conv (16->384)
                         da_spatial_fold=4,        # [NEW - da_adapter] the spatial fold and expansion factor r (4 for f32, 2 for f16)
                         stages_after_norm=False,  # [NEW - R2] moves the encoder stages and pairs it with the decoder M1 mirror
                         stages_norm_before_head=False,  # [NEW - R2n]
                         stages_after_head=False,  # [NEW - R3] encoder stages after the head conv; the decoder keeps R2's placement
                         stages_after_conv1=False,  # [NEW - R3a] stages after vae.conv1; the decoder keeps R2's placement
                         dec_stages_before_conv1=False,  # [NEW - R3a mirror] decoder stages in the z space (64ch) before conv1
                         decoder_mirror=True,      # [NEW - R2 control] False moves the encoder only and leaves the decoder as it is, so the parameter count does not change
                         expand_conv2=True,  # [NEW] True: conv2 z_dim→z_dim; False: conv2 z_dim→prior_z_dim (old)
                         expand_encoder_head=False,  # [NEW] True: encoder.head outputs z_dim*2 (instead of prior_z_dim*2)
                         use_b_adaptive=False,        # [NEW/B-fix]
                         b_adaptive_eps=1e-6,
                         b_adaptive_max=1e7,
                         b_adaptive_disc_weight=1.0,
                         **kwargs):
    # [NEW - da_adapter] the boundary adapter cannot reuse the pretrained head, conv1 or conv2 at
    #   all, since the channel counts differ, so it implies freshinit: force the pop and skip the
    #   copies. Variant b's decoder base conv does match in shape (384,16,3^3), so load_state_dict
    #   transplants it by itself and it is not popped.
    if da_adapter:
        zmain_fresh_init = True
    assert not (zmain_fresh_init and zmain_hybrid_init), '[init] fresh and hybrid are mutually exclusive'
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
        da_adapter=da_adapter,          # [NEW - da_adapter]
        da_base_split=da_base_split,
        da_spatial_fold=da_spatial_fold,
        stages_after_norm=stages_after_norm,  # [NEW - R2]
        stages_norm_before_head=stages_norm_before_head,  # [NEW - R2n]
        stages_after_head=stages_after_head,  # [NEW - R3]
        stages_after_conv1=stages_after_conv1,  # [NEW - R3a]
        dec_stages_before_conv1=dec_stages_before_conv1,  # [NEW - R3a mirror]
        decoder_mirror=decoder_mirror,        # [NEW - R2 control]
        use_b_adaptive=use_b_adaptive,
        b_adaptive_eps=b_adaptive_eps,
        b_adaptive_max=b_adaptive_max,
        b_adaptive_disc_weight=b_adaptive_disc_weight,
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
        # [NEW 2026-09-10 - single32] a single branch always has enc_out_dim = z_dim*2 (see :1309),
        #   so when z_dim != prior_z_dim the head[-1] size no longer matches the pretrained
        #   prior_z_dim*2. Without the pop, load_state_dict(strict=False) raises a RuntimeError on
        #   the size mismatch. With dual_branch=True the condition is exactly as before, so the
        #   existing lineages are unaffected.
        if expand_encoder_head or not dual_branch:
            # encoder.head[-1] output expands from prior_z_dim*2 to z_dim*2 → shape mismatch
            _zdim_keys += ['encoder.head.2.weight', 'encoder.head.2.bias']
        for k in _zdim_keys:
            if k in state and state[k].shape != model.state_dict().get(k, state[k]).shape:
                popped[k] = state.pop(k)

        # [freshinit] detach the residual chain (head.2/conv1/conv2) from the pretrained weights.
        #   At z16 (z_dim == prior_z_dim) the shapes match, so the mismatch-pop above does not catch
        #   them and load_state_dict would load them wholesale - hence the unconditional pop. The
        #   copy blocks below skip popped keys on the fresh branch, leaving the default random init
        #   from module construction (kaiming_uniform with a = sqrt(5)).
        #   conv2 is the exception: at z_dim == prior_z_dim it is shared with the base (z_prior) read
        #   chain, since conv2_prior is never built (see around line 1720), so randomising it would
        #   corrupt the base reconstruction. It keeps the pretrained weights.
        if (zmain_fresh_init or zmain_hybrid_init) and dual_branch:
            _fresh_keys = ['encoder.head.2.weight', 'encoder.head.2.bias',
                           'conv1.weight', 'conv1.bias']
            if z_dim != prior_z_dim:
                _fresh_keys += ['conv2.weight', 'conv2.bias']
            for k in _fresh_keys:
                if k in state:
                    popped[k] = state.pop(k)
            logging.info(f'[{"hybrid" if zmain_hybrid_init else "freshinit"}] popped from pretrained: {_fresh_keys}, '
                         f'gate_init={fresh_gate_init}')

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
                # [freshinit] on the fresh branch, skip the tile and grid copies entirely, so
                # head[-1] and conv1 keep the random init from construction
                if expand_encoder_head and not zmain_fresh_init and not zmain_hybrid_init:
                    # [2026-08-01 hybrid init, z48 from scratch] the new channels no longer start at
                    #   zero; they are tiled copies of the pretrained rows instead. At zero a new
                    #   channel emits only content-uncorrelated noise, so the decoder gate's expected
                    #   gradient is 0 - the swamp that cost 52k steps with no gain back at z32. Tiled,
                    #   the duplicated channels emit real content from step 0.
                    #   If z_dim % prior_z_dim != 0 the old zero behaviour stays, and at
                    #   z_dim == prior_z_dim the tile count is 1, which is the old behaviour too.
                    half = prior_z_dim
                    _rep = (z_dim // half) if (z_dim % half == 0) else 0
                    pre_ehw = popped.get('encoder.head.2.weight')
                    if pre_ehw is not None:
                        new_ehw = torch.zeros_like(model.encoder.head[-1].weight)
                        if _rep:
                            new_ehw[:z_dim] = pre_ehw[:half].repeat(_rep, *([1] * (pre_ehw.dim() - 1)))
                            new_ehw[z_dim:] = pre_ehw[half:].repeat(_rep, *([1] * (pre_ehw.dim() - 1)))
                        else:
                            new_ehw[:half, ...] = pre_ehw[:half, ...]                  # mu rows
                            new_ehw[z_dim:z_dim + half, ...] = pre_ehw[half:, ...]    # logvar rows
                        model.encoder.head[-1].weight.copy_(new_ehw)
                    pre_ehb = popped.get('encoder.head.2.bias')
                    if pre_ehb is not None:
                        new_ehb = torch.zeros_like(model.encoder.head[-1].bias)
                        if _rep:
                            new_ehb[:z_dim] = pre_ehb[:half].repeat(_rep)
                            new_ehb[z_dim:] = pre_ehb[half:].repeat(_rep)
                        else:
                            new_ehb[:half] = pre_ehb[:half]
                            new_ehb[z_dim:z_dim + half] = pre_ehb[half:]
                        model.encoder.head[-1].bias.copy_(new_ehb)
                    # conv1 (z_dim*2 -> z_dim*2): instead of zeros, tile a 2x2 (mu, logvar) block per
                    #   group. The pretrained conv1 (2h x 2h) contributes W_mm, W_ml, W_lm and W_ll
                    #   on the diagonal of each duplicated group, with zeros between groups.
                    pre_c1w = popped.get('conv1.weight')
                    pre_c1b = popped.get('conv1.bias')
                    if _rep and pre_c1w is not None and pre_c1w.shape[0] == 2 * half and pre_c1w.shape[1] == 2 * half:
                        W_mm, W_ml = pre_c1w[:half, :half], pre_c1w[:half, half:]
                        W_lm, W_ll = pre_c1w[half:, :half], pre_c1w[half:, half:]
                        new_c1 = torch.zeros_like(model.conv1.weight)
                        for _g in range(_rep):
                            mu_s = slice(_g * half, (_g + 1) * half)
                            lv_s = slice(z_dim + _g * half, z_dim + (_g + 1) * half)
                            new_c1[mu_s, mu_s] = W_mm
                            new_c1[mu_s, lv_s] = W_ml
                            new_c1[lv_s, mu_s] = W_lm
                            new_c1[lv_s, lv_s] = W_ll
                        model.conv1.weight.copy_(new_c1)
                        if model.conv1.bias is not None and pre_c1b is not None:
                            model.conv1.bias.copy_(torch.cat([pre_c1b[:half].repeat(_rep),
                                                              pre_c1b[half:].repeat(_rep)]))
                    else:
                        nn.init.zeros_(model.conv1.weight)
                        if model.conv1.bias is not None:
                            nn.init.zeros_(model.conv1.bias)

                # 1) conv1 (mu/log_var projection): (prior_z_dim*2 → prior_z_dim*2) → (prior_z_dim*2 → z_dim*2)
                #    input dim UNCHANGED (prior_z_dim*2=32), only output expands (32→64)
                #    pretrained rows: [0:prior_z_dim]=mu_old, [prior_z_dim:prior_z_dim*2]=logvar_old
                #    new layout after chunk(2): mu=[0:z_dim], logvar=[z_dim:z_dim*2]
                #    → mu_old rows go to [0:prior_z_dim], logvar_old rows go to [z_dim:z_dim+prior_z_dim]
                # [freshinit] skip the conv1 copy on the z16 path too: even though the fresh pop put
                # it in popped, it keeps the random init
                if 'conv1.weight' in popped and popped['conv1.weight'] is not None and not expand_encoder_head \
                        and not zmain_fresh_init:
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
                # [freshinit] on the fresh branch, skip the conv2 copy and keep the random init. This
                # is for z48 only: at z16 the sharing guard means conv2 is never popped, so
                # load_state_dict already loaded the pretrained weights and this block is not reached.
                if 'conv2.weight' in popped and popped['conv2.weight'] is not None and not zmain_fresh_init:
                    if expand_conv2:
                        # [2026-08-01 hybrid init] instead of zeros, tile the pretrained conv2 (h x h)
                        #   as diagonal blocks. All-zero here would stack on top of the zero decoder
                        #   gate and make a double-zero chain - the z32 swamp - so this delivers signal
                        #   as far as the gate. If z_dim % prior_z_dim != 0 the old zero behaviour
                        #   stays. Function preservation is handled by the zero z_main columns in
                        #   decoder.conv1.
                        _c2 = popped['conv2.weight']
                        _rep2 = (z_dim // prior_z_dim) if (z_dim % prior_z_dim == 0 and _c2.shape[0] == prior_z_dim) else 0
                        if _rep2:
                            new_c2 = torch.zeros_like(model.conv2.weight)
                            for _g in range(_rep2):
                                _s = slice(_g * prior_z_dim, (_g + 1) * prior_z_dim)
                                new_c2[_s, _s] = _c2
                            model.conv2.weight.data.copy_(new_c2)
                            if model.conv2.bias is not None and popped.get('conv2.bias') is not None:
                                model.conv2.bias.data.copy_(popped['conv2.bias'].repeat(_rep2))
                        else:
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
                    if zmain_fresh_init and fresh_gate_init == 'random':
                        # [freshinit -r] the residual columns keep the kaiming random init from
                        # construction; only the base columns are overwritten with pretrained weights below
                        new_dc1w = model.decoder.conv1.weight.detach().clone()
                    else:
                        new_dc1w = torch.zeros_like(model.decoder.conv1.weight)
                    if expand_conv2:
                        # z_main ch [0:z_dim]: stays zero(or fresh random); z_prior ch [z_dim:]: pretrained
                        new_dc1w[:, z_dim:, ...] = pre_dc1w
                    else:
                        if decoder_conv1_zmain_init == 'pretrained' and not zmain_fresh_init:
                            new_dc1w[:, :prior_z_dim, ...] = pre_dc1w   # z_main ch: pretrained copy
                        # else: z_main channels stay zero, or fresh random - fresh_gate_init alone decides
                        new_dc1w[:, prior_z_dim:, ...] = pre_dc1w       # z_prior ch: pretrained copy
                    model.decoder.conv1.weight.copy_(new_dc1w)
                    if 'decoder.conv1.bias' in popped and popped['decoder.conv1.bias'] is not None:
                        model.decoder.conv1.bias.copy_(popped['decoder.conv1.bias'])

                # [NEW - hybrid] z_main's first prior_z_dim channels reproduce the pretrained z16 path
                #   exactly, and the remaining new channels keep the kaiming random init from
                #   construction - neither tiled copies nor zeros.
                #   Why: in the z16 experiments a 1:1 inheritance beat random by 0.14 dB, and the z48
                #   tiling left the copied groups undifferentiated, at a correlation of 0.949.
                if zmain_hybrid_init and expand_encoder_head:
                    _h = prior_z_dim
                    _peh = popped.get('encoder.head.2.weight')
                    if _peh is not None:
                        model.encoder.head[-1].weight[:_h].copy_(_peh[:_h])                    # the first mu group
                        model.encoder.head[-1].weight[z_dim:z_dim + _h].copy_(_peh[_h:])       # the first logvar group
                    _pebv = popped.get('encoder.head.2.bias')
                    if _pebv is not None:
                        model.encoder.head[-1].bias[:_h].copy_(_pebv[:_h])
                        model.encoder.head[-1].bias[z_dim:z_dim + _h].copy_(_pebv[_h:])
                    # conv1: only the first group's 2x2 (mu, logvar) block is pretrained; the rest of
                    # that group's row is 0, which keeps the other groups from contaminating it
                    _pc1 = popped.get('conv1.weight')
                    if _pc1 is not None and _pc1.shape[0] == 2 * _h:
                        mu_s, lv_s = slice(0, _h), slice(z_dim, z_dim + _h)
                        model.conv1.weight[mu_s].zero_(); model.conv1.weight[lv_s].zero_()
                        model.conv1.weight[mu_s, mu_s] = _pc1[:_h, :_h]
                        model.conv1.weight[mu_s, lv_s] = _pc1[:_h, _h:]
                        model.conv1.weight[lv_s, mu_s] = _pc1[_h:, :_h]
                        model.conv1.weight[lv_s, lv_s] = _pc1[_h:, _h:]
                        _pc1b = popped.get('conv1.bias')
                        if model.conv1.bias is not None and _pc1b is not None:
                            model.conv1.bias[mu_s] = _pc1b[:_h]
                            model.conv1.bias[lv_s] = _pc1b[_h:]
                    # conv2: only the first group's row is pretrained, in the first group's columns;
                    # the rest of that row is 0, and the other groups' rows keep the random init
                    _pc2 = popped.get('conv2.weight')
                    if _pc2 is not None and expand_conv2 and _pc2.shape[0] == _h:
                        model.conv2.weight[:_h].zero_()
                        model.conv2.weight[:_h, :_h] = _pc2
                        _pc2b = popped.get('conv2.bias')
                        if model.conv2.bias is not None and _pc2b is not None:
                            model.conv2.bias[:_h] = _pc2b
                    logging.info(f'[hybrid] z_main: first {_h} channels inherited from pretrained, remaining {z_dim - _h} random')

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
            # With expand_encoder_head=True, encoder.head.2.weight/bias are popped and so go missing
            # from prior_state. prior_encoder.head[-1] always outputs prior_z_dim*2, which is the
            # pretrained shape, so it is restored from popped.
            # [freshinit] the fresh pop (z16 included) creates the same gap, and is restored the same way.
            if expand_encoder_head or zmain_fresh_init or zmain_hybrid_init:
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

        # ═══════════════════════════════════════════════════════════════════
        # [NEW 2026-09-10 - single32] initialisation for a single encoder, with no composite.
        #   This is written separately rather than reusing the dual block above, because the dual
        #   copy code assumes that z_prior's 16 channels are a frozen Wan, which pins half the latent
        #   at sigma ~= 0. A single branch has no such anchor, so the new channels' logvar is
        #   uncontrolled: measured, the 16 new channels averaged a logvar of -1 to +4 where the
        #   inherited ones sat at -65, and five of them were positive, giving sigma 1 to 5. During
        #   training latents_std then swung between 6 and 615 and the reconstruction loss would not
        #   come down, while the composite control under the same conditions held at 2.2.
        #
        #   The design, where the first _h = prior_z_dim channels reproduce the pretrained z16 path:
        #     [A] head[-1]  only the first mu/logvar group is pretrained. The new group's mu and
        #                   logvar rows keep their random init - zeroing them here would put them in
        #                   series with the zeros in conv1 below and no gradient would ever flow.
        #                   Only one of the two may be a zero gate.
        #     [B] conv1     the first group's row is the pretrained 2x2 block, with zeros in the
        #                   other groups' columns. The new mu rows keep their random init, and the
        #                   new logvar rows get zero weights with bias _LV0 - the fix for the
        #                   oscillation. sigma then starts at the constant e^(_LV0/2) and is learned
        #                   through the bias and weights.
        #     [C] conv2     only the first group's column in the first group's row is pretrained. The
        #                   new rows keep their random init; at zero the decoder's new columns would
        #                   receive no gradient and never open.
        #     [D] dec.conv1 the first group's columns are pretrained and the new columns are 0, a
        #                   no-op start. Note that zeroing the first columns as dual does would make
        #                   the whole decoder input 0 - a dead start.
        # ═══════════════════════════════════════════════════════════════════
        elif z_dim != prior_z_dim:
            _h = prior_z_dim
            _LV0 = -10.0        # sigma = e^(-5) ~= 0.0067; the inherited channels measure a logvar near -65
            with torch.no_grad():
                # [A] head[-1]
                _ph, _pb = popped.get('encoder.head.2.weight'), popped.get('encoder.head.2.bias')
                if _ph is not None:
                    model.encoder.head[-1].weight[:_h].copy_(_ph[:_h])                 # the first mu group
                    model.encoder.head[-1].weight[z_dim:z_dim + _h].copy_(_ph[_h:])    # the first logvar group
                if _pb is not None:
                    model.encoder.head[-1].bias[:_h].copy_(_pb[:_h])
                    model.encoder.head[-1].bias[z_dim:z_dim + _h].copy_(_pb[_h:])

                # [B] conv1
                _pc1, _pc1b = popped.get('conv1.weight'), popped.get('conv1.bias')
                _mu_s, _lv_s = slice(0, _h), slice(z_dim, z_dim + _h)
                if _pc1 is not None and _pc1.shape[0] == 2 * _h:
                    model.conv1.weight[_mu_s].zero_(); model.conv1.weight[_lv_s].zero_()
                    model.conv1.weight[_mu_s, _mu_s] = _pc1[:_h, :_h]
                    model.conv1.weight[_mu_s, _lv_s] = _pc1[:_h, _h:]
                    model.conv1.weight[_lv_s, _mu_s] = _pc1[_h:, :_h]
                    model.conv1.weight[_lv_s, _lv_s] = _pc1[_h:, _h:]
                    if model.conv1.bias is not None and _pc1b is not None:
                        model.conv1.bias[_mu_s] = _pc1b[:_h]
                        model.conv1.bias[_lv_s] = _pc1b[_h:]
                model.conv1.weight[z_dim + _h:].zero_()               # the new logvar rows
                if model.conv1.bias is not None:
                    model.conv1.bias[z_dim + _h:].fill_(_LV0)

                # [C] conv2
                _pc2, _pc2b = popped.get('conv2.weight'), popped.get('conv2.bias')
                if _pc2 is not None and _pc2.shape[0] == _h:
                    model.conv2.weight[:_h].zero_()
                    model.conv2.weight[:_h, :_h] = _pc2
                    if model.conv2.bias is not None and _pc2b is not None:
                        model.conv2.bias[:_h] = _pc2b

                # [D] decoder.conv1
                _pd, _pdb = popped.get('decoder.conv1.weight'), popped.get('decoder.conv1.bias')
                if _pd is not None:
                    _newd = torch.zeros_like(model.decoder.conv1.weight)
                    _newd[:, :_h, ...] = _pd
                    model.decoder.conv1.weight.copy_(_newd)
                    if _pdb is not None and model.decoder.conv1.bias is not None:
                        model.decoder.conv1.bias.copy_(_pdb)
            logging.info(f'[single32] first {_h} channels reproduce the pretrained z{_h} path / '
                         f'{z_dim - _h} new channels start with random mu and logvar {_LV0} / '
                         f'new decoder columns are 0, a no-op start')

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
