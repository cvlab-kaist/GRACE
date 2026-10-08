# [Source: Wan VAE] wan/modules/vae.py
# [Modified - oliviaa] Added extra encoder/decoder stages for higher compression.
# [NEW - oliviaa/skip] Skip connections on add_stages via channel averaging (DC-AE style).
# [NEW - oliviaa/geoprior] Dual-branch: upper (trainable) + lower (frozen prior encoder on 2x subsampled input).
#   Lower branch z is channel-concatenated with upper branch z before decoder.
#   decoder.conv1 expanded to accept (z_dim + prior_z_dim) input (pretrained in first prior_z_dim ch, zero rest).
#   Supports asymmetric z_dim: e.g. main z_dim=32, prior_z_dim=16 (Wan fixed) → decoder input 48ch.
#   subsample_mode: 'avg_pool' (default) | 'stride' | 'bilinear' — ablation-friendly.
# All modifications are marked with [NEW - oliviaa] or [Modified - oliviaa].
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import os
# [NEW 2026-07-20] VAE gc 확장 토글. ON 시 Resample(up/downsample conv) 도 gc(ResidualBlock 은 상시 gc).
#   grad+single-pass(feat_cache None) 경로에서만 checkpoint → VAE activation floor 축소로 nf↑ 가능.
#   dynamics 동일성 검증됨(verify_vae_gc_equiv.py: forward 비트동일, grad FP노이즈 수준). 기본 OFF.
_EXTENDED_VAE_GC = os.environ.get("KINEMADAE_EXTENDED_VAE_GC", "0") == "1"

__all__ = [
    'WanVAE',
]

CACHE_T = 2


# [NEW - oliviaa/skip] 3D pixel shuffle/unshuffle — from Open-Sora dc_ae/models/nn/vo_ops.py
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
                        'upsample2d_keepdim')  # [NEW - R2/M1] 폭 유지 공간 업샘플 (기존 upsample2d는 dim→dim//2)
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        # [NEW - R2/M1] 디코더 미러용: 채널 폭 유지 (384→384) — zero-init 시 skip(채널보존 nearest)만 남음
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
        # [NEW 2026-07-20] gc 확장: grad + single-pass(feat_cache None) 에서만 checkpoint.
        #   feat_idx 는 feat_cache!=None 브랜치에서만 뮤테이트되므로 no-cache recompute 는 안전.
        if _EXTENDED_VAE_GC and feat_cache is None and torch.is_grad_enabled():
            return checkpoint(self._forward_impl, x, feat_cache, feat_idx, use_reentrant=False)
        return self._forward_impl(x, feat_cache, feat_idx)

    def _forward_impl(self, x, feat_cache=None, feat_idx=[0]):
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

        # [Modified - oliviaa] downsample_temporal은 downsample3d와 temporal 로직 동일
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
                 add_stages=None,  # [NEW - oliviaa] list of {'mode': str, 'num_res_blocks': int}
                 da_adapter=False,        # [NEW - da_adapter] DA-VAE식 압축기: 스테이지 대신 채널압축(→z_dim/r²) + pixel_unshuffle(r)
                 da_spatial_fold=4,       # [NEW - da_adapter] 공간 접기 배율 r (f32t4=4, f16t4=2)
                 stages_after_norm=False,  # [NEW - R2] add_downsamples를 middle→RMS_norm→SiLU 뒤에서 실행 (mid 원해상 복귀)
                 stages_norm_before_head=False,  # [NEW - R2n] R2 스택 출력→head conv 사이 RMSnorm 재정규화 (스파이크 완화 1번)
                 stages_after_head=False,  # [NEW - R3] add_downsamples를 head conv **뒤**에서 실행 (폭 z_dim, head 도 원해상)
                 stages_after_conv1=False,  # [NEW - R3a] 스테이지를 여기서 **빌드만** 하고 실행은 GeopriorVAE.encode 가 conv1 뒤에서 함
                 use_b_adaptive=False,  # [NEW - oliviaa/B-fix] encoder.head[-1] 을 AdaptiveWeightedCausalConv3d 으로 교체 (= single backward path 의 weight gradient ratio mechanism)
                 b_adaptive_eps=1e-6,
                 b_adaptive_max=1e7,
                 b_adaptive_disc_weight=1.0):
        super().__init__()
        # [NEW - oliviaa/B-fix] flag 저장 — forward 에서 head 의 마지막 layer 의 두 output 처리 결정
        self.use_b_adaptive = use_b_adaptive
        # [NEW - da_adapter]
        self.da_adapter = da_adapter
        self.da_fold = da_spatial_fold
        # [NEW - R2]
        self.stages_after_norm = stages_after_norm
        assert not (da_adapter and stages_after_norm), "[R2/da] 구조 모드 동시 지정 불가"
        # [NEW - R2n] 스파이크 완화 1번: R2 는 34.5M 자유 스택이 norm·SiLU 뒤에 끼어 head conv 입력의
        #   정규화 앵커가 사라진다(7차 발산의 진앙 — grad_W 만 16배). 스택 출력을 RMSnorm 으로 재정규화해
        #   그 고리를 끊는다. norm 은 스케일만 묶고 채널 배열(align 자유도)은 보존. 기본 off = 비트 동일.
        self.stages_norm_before_head = stages_norm_before_head
        if stages_norm_before_head:
            assert stages_after_norm, "[R2n] stages_norm_before_head 는 R2(stages_after_norm) 전용"
        # [NEW - R3] 스테이지 위치는 셋 중 하나뿐이다(기본=middle 앞 / after_norm / after_head).
        #   둘 이상 켜지면 forward 의 분기 순서상 조용히 하나만 먹으므로 여기서 죽인다.
        self.stages_after_head = stages_after_head
        assert not (da_adapter and stages_after_head), "[R3/da] 구조 모드 동시 지정 불가"
        assert not (stages_after_norm and stages_after_head), \
            "[R3] stages_after_norm 과 stages_after_head 동시 지정 불가 (위치는 하나만)"
        # [NEW - R3a] 네 번째 위치. 스테이지는 conv1(96→96, 1x1) 출력과 폭이 같아야 하는데
        #   expand_encoder_head=True 면 enc_out_dim == z_dim*2 == conv1 출력이라 R3 와 같은 폭으로 빌드된다.
        #   (그 전제는 GeopriorVAE 쪽 assert 가 지킨다.) 실행은 여기서 하지 않는다 — encode() 가 한다.
        self.stages_after_conv1 = stages_after_conv1
        assert sum(bool(f) for f in (stages_after_norm, stages_after_head, stages_after_conv1)) <= 1, \
            "[R3a] 스테이지 위치 플래그는 하나만"
        assert not (da_adapter and stages_after_conv1), "[R3a/da] 구조 모드 동시 지정 불가"
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
        # [NEW - R3] 스테이지 폭. head conv 뒤에 놓이면 입력이 384(out_dim)가 아니라
        #   head 출력 폭 = z_dim(호출부가 넣는 enc_out_dim, 예: 48*2=96)이다.
        #   ※ pretrained_copy 는 384 짜리 사전학습 블록을 복사하므로 폭이 다르면 성립하지 않는다.
        _stage_dim = z_dim if (stages_after_head or stages_after_conv1) else out_dim
        assert not ((stages_after_head or stages_after_conv1) and any(
            (s.get('init') == 'pretrained_copy') for s in (add_stages or []))), \
            "[R3/R3a] 이 위치에서는 init='pretrained_copy' 불가 (사전학습 블록 폭 384 ≠ 스테이지 폭)"
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
        # [NEW - da_adapter] head conv = 채널 압축기 (384 → z_dim/r², 원해상). 공간 접기는 forward의
        #   pixel_unshuffle(r)이 담당 → 최종 z_dim 채널 @ /r 그리드 (latent shape 불변, mid는 원해상 유지).
        #   b_adaptive 는 이 conv에 그대로 장착 (head[-1] 자리 유지 → 트레이너 head 참조 무사)
        _head_out = z_dim
        if da_adapter:
            assert not add_stages, "[da_adapter] add_encoder_stages와 동시 사용 불가 (스테이지를 대체함)"
            assert z_dim % (da_spatial_fold ** 2) == 0, f"z_dim {z_dim} % r²={da_spatial_fold**2} != 0"
            _head_out = z_dim // (da_spatial_fold ** 2)
        # [NEW - oliviaa/B-fix] use_b_adaptive=True 시 마지막 conv = AdaptiveWeightedCausalConv3d (= (B) 식 의 weight gradient ratio mechanism)
        if use_b_adaptive:
            # local import — circular import 방지
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
        # [NEW - R2n] 재정규화 모듈 — off 면 미생성이라 state_dict 불변
        if self.stages_norm_before_head:
            self.stages_renorm = RMS_norm(out_dim, images=False)

    def _run_add_stages(self, x, feat_cache=None, feat_idx=[0], return_skip=False, ff_skips=None):
        # [NEW - R2] add_downsamples 블록 추출 — 호출 위치를 stages_after_norm 으로 선택.
        #   캐시-인지 코드/Option-B 청크동등 skip 그대로 (head 루프의 isinstance 분기에 태우면 캐시 누락됨)
        ## [NEW - oliviaa/skip] added downsample stages with skip connection
        for stage in self.add_downsamples:
            x_in = x
            B, C, T_in, H, W = x_in.shape
            for layer in stage:
                # [crossattn] add_downsamples Resample(downsample) 직전 feature = skip@32
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
        # [crossattn] return_skip=True 시 각 다운샘플 "전" feature 를 수집해
        #   (기존반환, [skip@256, skip@128, skip@64, skip@32]) 반환. False 면 기존과 100% 동일.
        #   skip 은 각 Resample(downsample) 직전의 x (해당 해상도 최고품질 feature).
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
            # [NEW 2026-07-21 gc-v2] full-res 첫 conv activation 도 checkpoint (Resample/Residual gc 와 동일 조건)
            if _EXTENDED_VAE_GC and torch.is_grad_enabled():
                x = checkpoint(self.conv1, x, use_reentrant=False)
            else:
                x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            # [crossattn] downsample Resample 직전 feature = 해당 해상도 skip
            if return_skip and isinstance(layer, Resample) and 'downsample' in layer.mode:
                ff_skips.append(x)
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## [NEW - oliviaa/skip] added downsample stages with skip connection
        # [NEW - R2] stages_after_norm=True 면 여기서 건너뛰고 middle→norm→SiLU 뒤에서 실행
        # [NEW - R3] stages_after_head=True 도 마찬가지로 건너뛴다. 이 가드를 빠뜨리면
        #   R3 에서 스테이지가 **여기서도 한 번 더** 돌아 384 입력이 96 폭 블록에 들어가 죽는다
        #   (실측: RuntimeError "size of tensor a (384) must match b (96)").
        # [NEW - R3a] stages_after_conv1 도 건너뛴다 — R3a 의 스테이지 실행 주체는 encode() 다.
        #   (R3 때 이 가드를 빠뜨려 스테이지가 두 번 돌아 384 vs 96 shape 에러가 났던 그 자리.)
        if not self.stages_after_norm and not self.stages_after_head and not self.stages_after_conv1:
            x = self._run_add_stages(x, feat_cache, feat_idx, return_skip, ff_skips)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        # [NEW - oliviaa/B-fix] use_b_adaptive=True 시 마지막 layer = AdaptiveWeightedCausalConv3d
        #                       → forward 시 (y_main, y_adv) tuple 반환 가능
        #                       → 호출 측 (= GeopriorVAE.encode) 에서 처리
        # [NEW - R2] stages_after_norm: norm·SiLU(사전학습 native 특징)를 지난 뒤 스테이지 실행, head conv만 남김
        if self.stages_after_norm:
            x = self.head[0](x)
            x = self.head[1](x)
            x = self._run_add_stages(x, feat_cache, feat_idx, return_skip, ff_skips)
            if self.stages_norm_before_head:
                x = self.stages_renorm(x)   # [NEW - R2n] 스택 출력 재정규화 → head conv 입력 앵커 복원
            _head_layers = [self.head[2]]
        else:
            _head_layers = self.head
        _da_pre = None
        for layer in _head_layers:
            # [NEW - da_adapter] 압축 conv 직전 feature 캡처 — 지름길은 conv와 같은 입력을 봄 (DA-VAE DADownBlock)
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
        # [NEW - oliviaa/B-fix] inference (= feat_cache is not None) 시 = backward 없음 → main only
        # training (= feat_cache is None) + use_b_adaptive=True 시 = tuple 그대로 → 호출 측 처리
        if isinstance(x, tuple) and feat_cache is not None:
            x = x[0]
        # [NEW - R3] stages_after_head: head conv 를 **다 지난 뒤** 스테이지 실행.
        #   목적 — head conv 입력이 SiLU(RMS_norm(·)) 로 되돌아가 묶인다(R2 는 스테이지 출력을 그대로
        #   받아 정규화가 없었고, 2026-08-13 발산에서 grad_W/grad_y 가 15배 튄 지점이 여기다).
        #   부수효과로 head conv 가 사전학습 해상도(16²)에서 돌고, 스테이지 폭이 384→z_dim 으로 준다.
        #   ※ b_adaptive training 시 x 는 (y_main, y_adv) tuple 이다. 두 갈래 모두 같은 스테이지를
        #     통과시켜야 한다 — da_adapter 가 바로 아래에서 쓰는 처리와 같은 이유.
        #     이때 스테이지 가중치는 두 경로의 기울기를 함께 받는다(R2 에서는 분기 이전이라 한 번만 받았다).
        if self.stages_after_head:
            if isinstance(x, tuple):
                x = tuple(self._run_add_stages(_x, feat_cache, feat_idx, False, None) for _x in x)
            else:
                x = self._run_add_stages(x, feat_cache, feat_idx, return_skip, ff_skips)
        # [NEW - da_adapter] 공간 접기 + 무가중치 지름길: unshuffle(384→384r²) 후 z_dim개 그룹평균.
        #   b_adaptive training 시 (y_main, y_adv) tuple — 두 출력 모두 동일 접기 (지름길 공유)
        if self.da_adapter:
            _r = self.da_fold
            _skip = _spatial_unshuffle3d(_da_pre, _r)
            _g = _skip.shape[1] // self.z_dim
            _skip = _skip.view(_skip.shape[0], self.z_dim, _g, *_skip.shape[2:]).mean(dim=2)
            if isinstance(x, tuple):
                x = tuple(_spatial_unshuffle3d(_x, _r) + _skip for _x in x)
            else:
                x = _spatial_unshuffle3d(x, _r) + _skip
        # [crossattn] return_skip=True 면 skip 리스트 함께 반환 (기존 x 는 그대로)
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
                 add_tail_stages=None,  # [NEW - oliviaa/geoprior] stages after head (3ch RGB space)
                 add_before_head_stages=None,  # [NEW - oliviaa/geoprior] upsample stages just before head (feature space)
                 da_adapter=False,      # [NEW - da_adapter] DA-VAE da_up식 진입 (conv→pixel_shuffle + repeat 지름길, mid 원해상)
                 da_base_split=False,   # [NEW - da_adapter] base(z_prior) 몫 분리: 사전학습 conv1(16→384) 유지 + nearest×r 합류
                 da_spatial_fold=4,     # [NEW - da_adapter] 공간 확대 배율 r
                 upsample_stages_before_middle=False,  # [NEW - R2/M1] add_upsamples를 middle 앞에서 실행 (미러)
                 upsample_stages_before_conv1=False,   # [NEW - R3a미러] add_upsamples를 **conv1 앞 z(64ch) 공간**에서 실행 — 사전학습 conv1 이 원해상에서 돎
                 dual_branch=False,   # [NEW - oliviaa/geoprior]
                 prior_z_dim=None,    # [NEW - oliviaa/geoprior] None → same as z_dim (symmetric)
                 expand_conv2=True,   # [NEW - oliviaa/geoprior] True: conv2 z_dim→z_dim → decoder input z_dim+prior_z_dim
                 first_frame_inject=False,  # [crossattn] 첫프레임 keyframe cross-attn 주입
                 ff_inject_levels="all",    # 주입할 upsample 레벨 ("all" 또는 "2,3" 등)
                 ff_window=32,              # windowed cross-attn window size
                 ff_encoder_source='residual',  # [crossattn base] cross-attn feature encoder: residual(self.encoder,4레벨) | base(prior_encoder,3레벨 @32없음)
                 ff_single_level=0,         # [crossattn single-level ablation] 0=레벨별 대칭(기본). 256/128/64/32 지정 시 그 한 encoder skip 을 전 디코더 레벨에 주입
                 ff_dual_source=False):     # [dual 2026-07-14] residual 4레벨 + base(prior_encoder) 3레벨(L1~L3) 동시 장착 (분리형 gated block 2세트)
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

        # [NEW - oliviaa/geoprior] dual_branch decoder input channels:
        #   expand_conv2=True:  conv2(z_dim→z_dim) + conv2_prior(prior_z_dim→prior_z_dim) → z_dim+prior_z_dim
        #   expand_conv2=False: conv2(z_dim→prior_z_dim) + conv2_prior(prior_z_dim→prior_z_dim) → prior_z_dim*2
        if dual_branch:
            _prior_z = prior_z_dim if prior_z_dim is not None else z_dim
            in_z = (z_dim + _prior_z) if expand_conv2 else (_prior_z * 2)
        else:
            _prior_z = None
            in_z = z_dim
        # [NEW - da_adapter] DA-VAE da_up식 진입: conv1(in_z→384)@/r 를 conv(→384r²)+pixel_shuffle 로 대체.
        #   da_base_split=True 면 base(z_prior) 몫은 사전학습과 같은 shape 의 conv1(16→384)로 분리 유지
        #   (state_dict 로드가 사전학습 dec.conv1 을 자동 이식) + nearest×r 로 32²에서 합류.
        self.da_adapter = da_adapter
        self.da_base_split = da_base_split and dual_branch
        self.da_fold = da_spatial_fold
        # [NEW - R2/M1]
        self.upsample_stages_before_middle = upsample_stages_before_middle
        assert not (da_adapter and upsample_stages_before_middle), "[R2/da] 구조 모드 동시 지정 불가"
        assert not (upsample_stages_before_middle and add_before_head_stages), \
            "[R2/M1] 미러는 before_head 스테이지와 동시 사용 불가 (제거하고 미러가 대체)"
        if da_adapter:
            assert not add_stages and not add_before_head_stages, \
                "[da_adapter] add_decoder(_before_head)_stages와 동시 사용 불가 (진입/mid 배치를 대체함)"
            _da_in = (in_z - _prior_z) if self.da_base_split else in_z
            _da_out = dims[0] * da_spatial_fold ** 2
            assert _da_out % _da_in == 0, f"repeat 지름길 불가: {_da_out} % {_da_in} != 0"
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

        # [NEW - oliviaa] Added upsample stages between middle and upsamples.
        # [NEW - R3a미러] conv1 앞 배치면 폭 = conv1 입력(in_z=64, z_main48+z_prior16).
        #   인코더 R3/R3a(사전학습 conv 뒤·z폭 스테이지 = +1.1~1.4dB 실측)의 디코더 판.
        #   위치 플래그는 하나만, da/pretrained_copy 불가, keepdim 모드만(확장형은 채널 2배라 conv1 과 어긋남).
        self.upsample_stages_before_conv1 = upsample_stages_before_conv1
        if upsample_stages_before_conv1:
            assert not upsample_stages_before_middle, "[R3a미러] 디코더 스테이지 위치는 하나만"
            assert not da_adapter, "[R3a미러/da] 동시 지정 불가"
            assert add_stages, "[R3a미러] add_decoder_stages 없이 켤 수 없음 (스테이지가 곧 업샘플러)"
            assert all(s.get('mode') == 'upsample2d_keepdim' for s in add_stages), \
                "[R3a미러] upsample2d_keepdim 모드만 지원 (확장형은 conv1 입력 폭과 어긋남)"
            assert not any(s.get('init') == 'pretrained_copy' for s in add_stages), \
                "[R3a미러] pretrained_copy 불가 (사전학습 블록 폭 384 ≠ 64)"
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
                # [NEW - R2/M1] 폭 유지 미러: ResBlock/Resample 모두 dim→dim (아래 else는 채널 2배 확장형)
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

        # [crossattn 2026-07] 첫프레임 keyframe cross-attn 주입 모듈.
        #   [변경] context 소스 = frozen VAE encoder return_skip 중간 feature (자작 pyramid 폐기).
        #   각 upsample 해상도 레벨의 "마지막 ResidualBlock 뒤" 1곳씩 주입 → 4-레벨 symmetric.
        #   주입점(coarse→fine): 각 level 의 마지막 resblock (다음 Resample 직전). Resample 로 level 경계 판단.
        #   query_dim = 그 지점 decoder feature ch (= dims[level+1]) — 실측: [384@32, 384@64, 192@128, 96@256].
        #   context_dim = encoder skip ch (해상도 1:1 대칭) — 실측: [384@32, 384@64, 192@128, 96@256].
        #   encoder skip 은 fine→coarse 순서 [skip@256(96), skip@128(192), skip@64(384), skip@32(384)] 로 수집됨
        #   → decoder level 0(coarse,@32) = skip idx 3, level 1(@64)=idx 2, level 2(@128)=idx 1, level 3(@256)=idx 0.
        self.first_frame_inject = first_frame_inject
        self.ff_encoder_source = ff_encoder_source  # [NEW base] residual | base
        self.ff_single_level = int(ff_single_level)  # [NEW single-level] 0=off, 256/128/64/32=단일 skip 전 레벨 주입
        # [NEW single-level ablation] encoder skip 은 fine→coarse 수집: idx0=@256(96ch), idx1=@128(192ch),
        #   idx2=@64(384ch), idx3=@32(384ch). res → (skip_idx, channels) 매핑.
        _single_map = {256: (0, 96), 128: (1, 192), 64: (2, 384), 32: (3, 384)}
        self.ff_level_channels = dims[1:]         # [384,384,192,96] — decoder level(coarse→fine) out_ch
        self.ff_inject = nn.ModuleDict()          # key=str(upsamples flat idx), val=GatedCrossAttnBlock
        self.ff_inject_skip = {}                  # flat idx -> encoder skip 리스트 index (fine→coarse 순서 기준)
        if first_frame_inject:
            from crossattn_ff import GatedCrossAttnBlock
            _nlv = len(dim_mult)                  # 4 레벨
            if str(ff_inject_levels).lower() == "all":
                _lv = set(range(_nlv))
            else:
                _lv = set(int(s) for s in str(ff_inject_levels).split(",") if s.strip() != "")
            # 각 upsample 레벨의 "마지막 ResidualBlock" flat idx 계산 (다음 Resample 직전 / 마지막 레벨은 끝).
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
                # [NEW base] base encoder(prior_encoder)는 add_downsample 없어 최심 @32 skip 없음
                #   → decoder coarse level 0(@32) 주입 생략(3레벨만). skip_idx 매핑(level 1,2,3→2,1,0)은
                #   base skip 리스트 [256,128,64](idx0,1,2)에 그대로 맞음(@64→2,@128→1,@256→0).
                if ff_encoder_source == 'base' and _level == 0:
                    continue
                _flat = _last_resblock_of_level[_level]
                _out = dims[_level + 1]           # query dim = decoder feature ch @ this level (레벨별 유지)
                if self.ff_single_level and self.ff_single_level in _single_map:
                    # [NEW single-level ablation] 지정한 단일 encoder skip 을 모든 디코더 레벨에 주입.
                    #   query dim(_out)은 레벨별 그대로, context 만 단일 skip(고정 idx/ch)으로 통일.
                    #   resolution mismatch(디코더 레벨 ≠ skip 해상도)는 CrossAttention 의 window 스케일링이 흡수.
                    _skip_idx, _ctx = _single_map[self.ff_single_level]
                else:
                    # 기본: 레벨별 대칭. encoder skip index: fine→coarse 수집이라 coarse level 0 = skip idx (nlv-1)
                    _skip_idx = _nlv - 1 - _level
                    # context dim = encoder skip ch @ matched resolution (실측: L0=384,L1=384,L2=192,L3=96).
                    _ctx = dims[_level + 1]
                self.ff_inject[str(_flat)] = GatedCrossAttnBlock(
                    dim=_out, context_dim=_ctx, window_size=(ff_window, ff_window))
                self.ff_inject_skip[_flat] = _skip_idx

        # [dual 2026-07-14] dual-source: residual(위) + base(prior_encoder) 블록 동시 장착.
        #   base 인코더는 add_downsample 이 없어 skip 3개(fine→coarse [@256,@128,@64]) → L0 제외 L1~L3만.
        #   매핑/채널은 기존 ff_encoder_source='base' 경로와 동일 (level→skip idx = nlv-1-level, ctx=dims[level+1]).
        #   γ=0 init 이라 장착 시점 출력 불변. forward 는 residual 주입 뒤 순차 합산.
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
            for _level in range(1, _nlv):            # base 는 @32 skip 없음 → L0 제외
                if _level not in _last_rb:
                    continue
                _flat = _last_rb[_level]
                _out = dims[_level + 1]
                self.ff_inject_base[str(_flat)] = _GCB(
                    dim=_out, context_dim=dims[_level + 1], window_size=(ff_window, ff_window))
                self.ff_inject_base_skip[_flat] = _nlv - 1 - _level

        # [NEW - oliviaa/geoprior] add_before_head: DC-AE style upsample just before head (out_dim feature space)
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

        # [NEW - oliviaa] Deferred pretrained_copy init for add_upsamples
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

        # [NEW - oliviaa/geoprior] add_tail: stages after head in 3ch RGB space
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
        # [NEW - R2/M1] add_upsamples 블록 추출 — 호출 위치를 upsample_stages_before_middle 로 선택
        ## [NEW - oliviaa/skip] added upsample stages with skip connection
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
        # [NEW - da_adapter] CausalConv3d 캐시 적용 공통화 — 기존 conv1 블록과 동일 로직
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
        ## conv1  ([NEW - da_adapter] da_up 진입 분기)
        if self.da_adapter:
            # cat 순서 = [z_main(0:z_dim), z_prior(z_dim:)] (WanVAE_.decode)
            if self.da_base_split:
                xm, xb = x[:, :self._da_in], x[:, self._da_in:]
            else:
                xm, xb = x, None
            h = self._cached_conv(self.da_up, xm, feat_cache, feat_idx)
            h = _spatial_shuffle3d(h, self.da_fold)
            # 무가중치 지름길 (DA-VAE DAUpBlock): 채널 반복 → shuffle
            _sc = xm.repeat_interleave(self._da_repeats, dim=1)
            h = h + _spatial_shuffle3d(_sc, self.da_fold)
            if xb is not None:
                # base 경로: 사전학습 conv1(16→384)@/r → nearest×r 로 원해상 합류
                b = self._cached_conv(self.conv1, xb, feat_cache, feat_idx)
                _B = b.shape[0]
                b = rearrange(b, 'b c t h w -> (b t) c h w')
                b = F.interpolate(b, scale_factor=self.da_fold, mode='nearest')
                b = rearrange(b, '(b t) c h w -> b c t h w', b=_B)
                h = h + b
            x = h
        else:
            # [NEW - R3a미러] 스테이지를 conv1 **앞** z(64ch) 공간에서 실행 → conv1 이 원해상에서 돎.
            #   zero-init 시 스텝 0 = repeat+pixel_shuffle skip(무가중치 nearest류 업샘플) 만 남음.
            if self.upsample_stages_before_conv1:
                x = self._run_add_upsamples(x, feat_cache, feat_idx)
            x = self._cached_conv(self.conv1, x, feat_cache, feat_idx)

        # [NEW - R2/M1] 미러: 업샘플 스테이지를 middle 앞에 실행 → mid가 원해상(32²)에서 돎
        if self.upsample_stages_before_middle:
            x = self._run_add_upsamples(x, feat_cache, feat_idx)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## [NEW - oliviaa/skip] added upsample stages with skip connection
        # [NEW - R3a미러] before_conv1 이면 이미 돌았으므로 여기(legacy after-middle)서도 건너뛴다
        #   — 인코더 R3 때 이 가드를 빠뜨려 이중실행 shape 에러가 났던 것과 같은 자리.
        if not self.upsample_stages_before_middle and not self.upsample_stages_before_conv1:
            x = self._run_add_upsamples(x, feat_cache, feat_idx)

        ## upsamples  (+ [crossattn] 각 ResidualBlock 뒤 keyframe cross-attn 주입)
        for _u, layer in enumerate(self.upsamples):
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
            # gate(gamma)=0 시작이라 ff_skips 있어도 첫 forward 는 bit-identical. feat_cache 미사용(AttentionBlock 패턴).
            #   ff_skips[i] = encoder skip (2D, B,C,H',W'), 해상도 1:1 대칭 → cross-attn context.
            if self.first_frame_inject and ff_skips is not None and str(_u) in self.ff_inject:
                x = self.ff_inject[str(_u)](x, ff_skips[self.ff_inject_skip[_u]])
            # [dual] base(prior_encoder) 소스 블록 — residual 주입 뒤 순차 합산.
            #   x = x + γ_res·attn_res(...) 다음에 x = x + γ_base·attn_base(...). γ=0 init → 장착 무영향.
            if self.first_frame_inject and ff_skips_base is not None and str(_u) in self.ff_inject_base:
                x = self.ff_inject_base[str(_u)](x, ff_skips_base[self.ff_inject_base_skip[_u]])

        ## [NEW - oliviaa/geoprior] add_before_head: DC-AE style upsample in feature space (before head)
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
                    # [FIX 2026-06-27 frame-drop root cause] chunked decode: main Resample은
                    #   chunk당 full 2x temporal(2t) 생산. 기존엔 이 chunk 분기가 없어
                    #   single-pass 로직(1+(t-1)*2 = 2t-1)을 chunked 에도 적용 → main(2t)과
                    #   1프레임 어긋나 x+skip 에서 손실(17->15, 81->71). add_upsamples skip
                    #   (line 744-752)과 동일하게 chunk 분기 추가해 길이 일치.
                    if T_in == 1:
                        # first chunk: Resample 'Rep'(temporal 유지) — 공간만 2x
                        skip = rearrange(x_in, 'b c t h w -> (b t) c h w')
                        skip = skip.repeat_interleave(4, dim=1)
                        skip = F.pixel_shuffle(skip, 2)
                        skip = rearrange(skip, '(b t) c h w -> b c t h w', b=B)
                    else:
                        # subsequent chunk: full 2x temporal(2t) — main 과 길이 일치
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
                    # [FIX] chunked: main 과 동일하게 full 2x temporal
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
        # [NEW 2026-07-21 gc-v2] full-res head 도 checkpoint (feat_cache None + grad 시).
        #   early-return 금지 — head 뒤 add_tail 단계 보존 (gc 시 아래 루프는 빈 순회).
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

        ## [NEW - oliviaa/geoprior] add_tail: DC-AE style refinement in 3ch RGB space
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


# [NEW - da_adapter] 공간 전용 shuffle/unshuffle (temporal 불변, 채널-major: out_ch = c*r² + phase)
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
                 add_decoder_tail_stages=None,          # [NEW - oliviaa/geoprior] stages after head (3ch RGB space)
                 add_decoder_before_head_stages=None,   # [NEW - oliviaa/geoprior] upsample stages just before head
                 dual_branch=False,           # [NEW - oliviaa/geoprior]
                 subsample_mode='avg_pool',   # [NEW - oliviaa/geoprior] 'avg_pool' | 'stride' | 'bilinear'
                 prior_z_dim=None,            # [NEW - oliviaa/geoprior] None → same as z_dim; int for asymmetric (e.g. 16 for frozen Wan)
                 expand_conv2=True,           # [NEW - oliviaa/geoprior] True: conv2 z_dim→z_dim; False: conv2 z_dim→prior_z_dim (old)
                 expand_encoder_head=False,   # [NEW - oliviaa/geoprior] True: encoder.head outputs z_dim*2 (instead of prior_z_dim*2)
                 use_b_adaptive=False,        # [NEW - oliviaa/B-fix] encoder.head[-1] = AdaptiveWeightedCausalConv3d (= single backward weight ratio)
                 b_adaptive_eps=1e-6,
                 b_adaptive_max=1e7,
                 b_adaptive_disc_weight=1.0,
                 da_adapter=False,            # [NEW - da_adapter] DA-VAE식 경계 어댑터 (인코더 채널압축+unshuffle / 디코더 da_up)
                 da_base_split=False,         # [NEW - da_adapter] 디코더 base 몫 분리 보존 (b′)
                 da_spatial_fold=4,           # [NEW - da_adapter] 공간 접기/확대 배율 r
                 stages_after_norm=False,     # [NEW - R2] 인코더 스테이지 norm·SiLU 뒤 + 디코더 M1 미러 (한 플래그 패키지)
                 stages_norm_before_head=False,  # [NEW - R2n]
                 stages_after_head=False,     # [NEW - R3] 인코더 스테이지 head conv 뒤. 디코더는 R2 와 동일 배치 유지
                 stages_after_conv1=False,    # [NEW - R3a] 스테이지를 vae.conv1 뒤(chunk 직전). 사전학습 head→conv1 이 원조합·원해상으로 붙음
                 dec_stages_before_conv1=False,  # [NEW - R3a미러] 디코더 스테이지를 conv1 앞 z(64ch)로 — 사전학습 dec.conv1 이 원해상에서 돎
                 decoder_mirror=True,         # [NEW - R2 통제] False = 인코더만 이동
                 first_frame_inject=False,    # [crossattn] 첫프레임 keyframe cross-attn 주입
                 ff_inject_levels="all",      # 주입할 upsample 레벨
                 ff_window=32,                # windowed cross-attn window size
                 ff_encoder_source='residual',  # [crossattn base] residual(self.encoder) | base(prior_encoder full-res)
                 ff_single_level=0,           # [crossattn single-level ablation] 0=off, 256/128/64/32=단일 skip 전 레벨 주입
                 ff_dual_source=False):       # [dual 2026-07-14] residual+base 참조 동시 장착
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
        # [NEW - oliviaa/geoprior] prior branch z_dim (Wan fixed = 16); None → symmetric
        self.prior_z_dim = prior_z_dim if prior_z_dim is not None else z_dim

        self.expand_encoder_head = expand_encoder_head
        # modules
        # [NEW - oliviaa/geoprior] dual_branch: encoder head fixed at prior_z_dim*2 (same as pretrained Wan)
        # → encoder.head[-1] keeps shape (384→prior_z_dim*2), no weight mismatch
        # expand_encoder_head=True: encoder.head outputs z_dim*2 (true channel expansion through head)
        enc_out_dim = (self.prior_z_dim * 2) if (dual_branch and not expand_encoder_head) else (z_dim * 2)
        # [NEW - da_adapter] 플래그 저장 + main encoder/decoder 에만 전달 (prior_encoder 는 순정 유지)
        self.da_adapter = da_adapter
        self.da_base_split = da_base_split
        self.da_spatial_fold = da_spatial_fold
        self.stages_after_norm = stages_after_norm  # [NEW - R2]
        self.stages_after_head = stages_after_head  # [NEW - R3]
        self.stages_after_conv1 = stages_after_conv1  # [NEW - R3a]
        # [NEW - R3a] 스테이지 폭은 Encoder3d 가 자기 z_dim(=enc_out_dim)으로 짓는다. conv1 출력은
        #   z_dim*2 이므로, 둘이 같으려면 expand_encoder_head 가 필수다(그때 enc_out_dim == z_dim*2).
        #   아니면 conv1 출력(96)과 스테이지 폭(32)이 어긋나 첫 forward 에서 죽는다 — 여기서 미리 죽인다.
        if stages_after_conv1:
            assert expand_encoder_head, \
                "[R3a] stages_after_conv1 은 expand_encoder_head=True 전제 (스테이지 폭 == conv1 출력 폭)"
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
                # [NEW - oliviaa/geoprior] separate frozen conv2 for prior branch
                self.conv2_prior = CausalConv3d(self.prior_z_dim, self.prior_z_dim, 1)
        else:
            self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout,
                                 add_stages=add_decoder_stages,
                                 add_tail_stages=add_decoder_tail_stages,
                                 add_before_head_stages=add_decoder_before_head_stages,  # [NEW - oliviaa/geoprior]
                                 da_adapter=da_adapter,          # [NEW - da_adapter]
                                 da_base_split=da_base_split,
                                 da_spatial_fold=da_spatial_fold,
                                 # [NEW - R3] stages_after_head 도 포함시킨다. 안 그러면 R3 를 켤 때
                                 #   이 값이 False 로 떨어져 **디코더가 조용히 'middle 뒤'로 바뀐다**
                                 #   (= R2 대비 인코더·디코더 둘 다 달라져 단일 변인이 깨짐).
                                 #   R3 의 의도는 "인코더만 이동, 디코더는 R2 그대로" 이다.
                                 # [NEW - R3a미러] before_conv1 이 켜지면 before_middle 은 꺼야 한다(위치 하나만).
                                 #   안 끄면 Decoder3d assert 로 죽음 — 조용한 이중배치 방지.
                                 upsample_stages_before_middle=((stages_after_norm or stages_after_head or stages_after_conv1) and decoder_mirror and not dec_stages_before_conv1),  # [NEW - R2/M1 패키지, R3/R3a 도 디코더는 R2 배치 유지]
                                 upsample_stages_before_conv1=dec_stages_before_conv1,  # [NEW - R3a미러]
                                 dual_branch=dual_branch,
                                 prior_z_dim=self.prior_z_dim,   # [NEW - oliviaa/geoprior]
                                 expand_conv2=expand_conv2,       # [NEW - oliviaa/geoprior]
                                 first_frame_inject=first_frame_inject,  # [crossattn]
                                 ff_inject_levels=ff_inject_levels,
                                 ff_window=ff_window,
                                 ff_encoder_source=ff_encoder_source,  # [crossattn base]
                                 ff_single_level=ff_single_level,  # [crossattn single-level]
                                 ff_dual_source=ff_dual_source)   # [dual]

        # [crossattn 2026-07] 첫프레임 keyframe feature 는 frozen VAE encoder 의 return_skip
        #   중간 feature 로 대체 → 별도 ff_encoder 없음(새 param 0). decode() 에서 self.encoder 재사용.

        # [NEW - oliviaa/geoprior] 하단 브랜치: vanilla Wan encoder (add_stages 없음, frozen)
        # prior branch uses prior_z_dim (Wan's fixed z_dim=16); weights copied in _video_vae_geoprior
        if dual_branch:
            self.prior_encoder = Encoder3d(dim, self.prior_z_dim * 2, dim_mult, num_res_blocks,
                                           attn_scales, self.temperal_downsample, dropout,
                                           add_stages=None)
            # separate projection for prior branch (conv1 handles main encoder only)
            self.prior_conv1 = CausalConv3d(self.prior_z_dim * 2, self.prior_z_dim * 2, 1)

        # [NEW - oliviaa] cache count_conv3d results — avoids full module-tree traversal every step
        self._cached_dec_conv_num = count_conv3d(self.decoder)
        self._cached_enc_conv_num = count_conv3d(self.encoder)
        if dual_branch:
            self._cached_prior_conv_num = count_conv3d(self.prior_encoder)

    def forward(self, x):
        # [NEW - oliviaa/B-fix] encode 의 return 가 tuple ((mu, log_var), (mu_adv, log_var_adv)) 가능
        # 단 GeopriorVAE.forward 은 일반 reconstruction path — main branch (= first) 만 사용.
        # 호출 측 (= GeopriorDiTAlignModel.forward 등) 에서 use_b_adaptive 시 adv branch 별도 처리.
        encode_result = self.encode(x, scale=None)
        if isinstance(encode_result[0], tuple):
            (mu, log_var), _ = encode_result    # adv branch 무시 (= 일반 forward path)
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
        # [NEW - R3a] 스테이지를 conv1 **뒤**·chunk **앞**에서 실행.
        #   · conv1 이 32²(원해상)에서 돌게 되고, 사전학습 head→conv1 이 원래 조합 그대로 붙는다.
        #     (conv1 은 1x1 이라 해상도 무관 — 이 이동의 기대효과는 '재매개변수화 수준'이며,
        #      이 런의 목적은 그 예측 자체의 검증이다.)
        #   · 이 지점은 청크 결합(torch.cat) **뒤**라, 스테이지의 causal conv 가 full-T 를 한 번에 본다
        #     → feat_cache 불요, chunked/single-pass 등가가 구조적으로 성립 (R2/R3 의 청크등가 로직 불필요).
        #   · b_adaptive tuple 이면 두 갈래를 같은 스테이지(가중치 공유)에 각각 태운다 — R3 와 동일 규약.
        def _r3a(y):
            return self.encoder._run_add_stages(y, None, [0]) if self.stages_after_conv1 else y
        # [NEW - oliviaa/B-fix] out 가 tuple (y_main, y_adv) 가능 — use_b_adaptive=True + training 시
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

    # [NEW - oliviaa/geoprior] 하단 브랜치 인코딩
    # subsample_mode에 따라 2x T+H+W 다운샘플 → frozen prior_encoder → mu_prior
    def _encode_prior(self, x):
        if self.subsample_mode == 'avg_pool':
            # CausalVAE는 첫 프레임을 따로 처리 → T가 홀수여야 latent T가 맞음
            # avg_pool3d는 floor(T/2)를 만드므로 T가 홀수면 마지막 프레임 repeat해서 짝수로 맞춤
            if x.shape[2] % 2 == 1:
                x_pad = torch.cat([x, x[:, :, -1:, :, :]], dim=2)
            else:
                x_pad = x
            x_sub = F.avg_pool3d(x_pad, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        elif self.subsample_mode == 'spatial_avg_temporal_stride':
            # [NEW - oliviaa] spatial 2x avg + temporal stride 2 (no temporal averaging)
            # I2V 첫 프레임 값 보존을 위해 temporal averaging 제거. GT frame은 절반만 사용.
            # T가 홀수면 마지막 프레임 repeat해서 짝수로 맞춤 (output T는 'avg_pool'과 동일)
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
        # [NEW - oliviaa/f32t4] prior subsample 을 main add_stage 와 짝맞춤. temporal 미압축
        #   (add_encoder 가 downsample2d = spatial-only 이므로 prior 도 spatial 만 줄이고 temporal 유지).
        #   → main·prior latent 크기 일치 → concat 성립. base(×8/×4) 통과 후:
        #   bilinear_s2t1 → spatial ×16, temporal ×4 (f16t4) / bilinear_s4t1 → spatial ×32, temporal ×4 (f32t4).
        elif self.subsample_mode == 'bilinear_s2t1':
            B, C, T, H, W = x.shape
            x_flat = rearrange(x, 'b c t h w -> (b t) c h w')  # temporal 미압축 (::2 제거)
            x_flat = F.interpolate(x_flat, scale_factor=0.5, mode='bilinear', align_corners=False)  # spatial ×2
            x_sub = rearrange(x_flat, '(b t) c h w -> b c t h w', b=B)
        elif self.subsample_mode == 'bilinear_s4t1':
            B, C, T, H, W = x.shape
            x_flat = rearrange(x, 'b c t h w -> (b t) c h w')  # temporal 미압축
            x_flat = F.interpolate(x_flat, scale_factor=0.25, mode='bilinear', align_corners=False)  # spatial ×4 (한번에)
            x_sub = rearrange(x_flat, '(b t) c h w -> b c t h w', b=B)
        # [NEW 2026-08-02 AA] ×0.5 두 번 = 안티앨리어싱 4배 축소. 근거: s4t1 한방은 HF(앨리어스) 2배 실측,
        #   학습된 s4t1 ckpt 에 eval 스왑만으로 +0.37dB (8/8 개선) — 앨리어싱이 f32 벽(법칙 대비 −2dB)의 유력 지분.
        elif self.subsample_mode == 'bilinear_s4t1_2stage':
            B, C, T, H, W = x.shape
            x_flat = rearrange(x, 'b c t h w -> (b t) c h w')  # temporal 미압축
            x_flat = F.interpolate(x_flat, scale_factor=0.5, mode='bilinear', align_corners=False)
            x_flat = F.interpolate(x_flat, scale_factor=0.5, mode='bilinear', align_corners=False)
            x_sub = rearrange(x_flat, '(b t) c h w -> b c t h w', b=B)
        else:
            raise ValueError(f'Unknown subsample_mode: {self.subsample_mode}')

        t = x_sub.shape[2]
        # prior_encoder는 add_downsamples 없음 → base tf만
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
        # [crossattn 2026-07] 첫프레임 keyframe → frozen VAE encoder 중간 feature(return_skip).
        #   chunk 무관, 한 번만 인코딩. first_frame None(=T2V/ff_drop) 또는 inject off 면 None → 기존 경로.
        #   encoder 는 CausalConv3d(temporal) 이라 T>=3 필요 → [첫프레임 + zeros(T-1)] 로 인코딩.
        #   각 skip 은 5D(B,C,T',H,W) → temporal idx 0(첫프레임) 슬라이스 = 2D context.
        ff_skips = None
        ff_skips_base = None   # [dual]
        if first_frame is not None and getattr(self, 'first_frame_inject', False):
            # single-pass encoder 는 temporal downsample 마다 time_conv(kernel3,no-pad) 를 거쳐
            #   T 를 줄이며 각 단계 입력 T>=3 필요. temporal downsample n 개 → 안전 T = 4n+1.
            #   (실측: 이 geoprior config 3개 → T=9 에서 통과, skip 들의 temporal idx 0 은 항상 첫프레임)
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
            # [crossattn base] feature encoder 선택:
            #   residual(기본): self.encoder (add_downsample 포함 4레벨 skip)
            #   base: self.prior_encoder (frozen pure Wan, add_downsample 없어 3레벨 skip @256/128/64).
            #         full-res _ff5 그대로 통과(다운샘플 안 함). @32 없음 → decoder @32 주입은 __init__에서 이미 제외.
            if getattr(self, 'ff_encoder_source', 'residual') == 'base' and hasattr(self, 'prior_encoder'):
                _, _skips5 = self.prior_encoder(_ff5, return_skip=True)
            else:
                _, _skips5 = self.encoder(_ff5, return_skip=True)   # single-pass (feat_cache 없음)
            # 각 skip 의 temporal idx 0 = 첫프레임 → 2D (B,C,H',W')
            ff_skips = [s[:, :, 0] for s in _skips5]
            # [dual] dual-source: base(prior_encoder, full-res, frozen pure Wan) skip 추가 수집.
            #   기존 'base' 단독 경로와 동일 호출 — 여기선 residual 과 병행. L1~L3 블록이 소비.
            if getattr(self, 'ff_dual_source', False) and hasattr(self, 'prior_encoder'):
                _, _skips5b = self.prior_encoder(_ff5, return_skip=True)
                ff_skips_base = [s[:, :, 0] for s in _skips5b]
        # [NEW - oliviaa/geoprior] dual_branch: conv2 for z_main; conv2_prior (frozen) for z_prior
        if self.dual_branch:
            z_main = self.conv2(z[:, :self.z_dim])              # (B, prior_z_dim, T', H', W')
            if self.z_dim == self.prior_z_dim:
                z_p = self.conv2(z[:, self.z_dim:])             # shared conv2
            else:
                z_p = self.conv2_prior(z[:, self.z_dim:])       # separate frozen conv2_prior
            x = torch.cat([z_main, z_p], dim=1)                 # (B, prior_z_dim*2, T', H', W')
        else:
            x = self.conv2(z)
        # [NEW] force_single_pass: eval-mode 에서도 chunk 안 하고 single-pass decode (81f chunk 버그 회피용).
        #   chunking 은 긴 영상 메모리 절약용이라 기본 유지, 정확한 측정 필요 시에만 플래그 켬.
        # [NEW group-chunk] decode_chunk_latent>0: K latent position 씩 묶어 chunked decode (학습+eval 모두).
        #   목적: 256x81 학습 시 single-pass upsample backward INT_MAX 회피하면서 frame-by-frame(느림) 안 쓰기.
        #   feat_cache가 경계 context 이어주므로 single-pass와 동치(검증 필수). default 0 = 기존 동작 유지.
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


# [NEW - oliviaa/geoprior] dual-branch VAE loader
def _video_vae_geoprior(pretrained_path=None, z_dim=None, device='cpu',
                         add_encoder_stages=None, add_decoder_stages=None,
                         add_decoder_tail_stages=None,          # [NEW] stages after head (3ch RGB space)
                         add_decoder_before_head_stages=None,   # [NEW] upsample stages just before head
                         dual_branch=True, subsample_mode='avg_pool',
                         prior_z_dim=16,  # [NEW] fixed Wan z_dim for prior branch
                         decoder_conv1_zmain_init='zero',  # [NEW] 'zero' or 'pretrained' for z_main ch
                         zmain_fresh_init=False,   # [NEW - freshinit] residual 사슬(head.2/conv1/conv2) 사전학습-카피 제거 → 생성 시 기본 랜덤 유지
                         fresh_gate_init='zero',   # [NEW - freshinit] decoder.conv1 residual 열: 'zero' | 'random' (zmain_fresh_init=True일 때만 의미)
                         zmain_hybrid_init=False,  # [NEW - hybrid] z_main 첫 prior_z_dim 채널만 사전학습 승계, 나머지 신규 채널은 랜덤
                         da_adapter=False,         # [NEW - da_adapter] DA-VAE식 경계 어댑터 (b/b′) — zmain_fresh_init 자동 함의
                         da_base_split=False,      # [NEW - da_adapter] b′: 디코더 base 몫을 사전학습 conv(16→384)로 분리 보존
                         da_spatial_fold=4,        # [NEW - da_adapter] 공간 접기/확대 배율 r (f32=4, f16=2)
                         stages_after_norm=False,  # [NEW - R2] 인코더 위치 이동 + 디코더 M1 미러 패키지
                         stages_norm_before_head=False,  # [NEW - R2n]
                         stages_after_head=False,  # [NEW - R3] 인코더 스테이지를 head conv 뒤로 (디코더는 R2 유지)
                         stages_after_conv1=False,  # [NEW - R3a] 스테이지를 vae.conv1 뒤로 (디코더는 R2 유지)
                         dec_stages_before_conv1=False,  # [NEW - R3a미러] 디코더 스테이지를 conv1 앞 z(64ch)로
                         decoder_mirror=True,      # [NEW - R2 통제] False 면 인코더만 이동(디코더 현행 유지) → 파라미터 변화 0
                         expand_conv2=True,  # [NEW] True: conv2 z_dim→z_dim; False: conv2 z_dim→prior_z_dim (old)
                         expand_encoder_head=False,  # [NEW] True: encoder.head outputs z_dim*2 (instead of prior_z_dim*2)
                         use_b_adaptive=False,        # [NEW - oliviaa/B-fix]
                         b_adaptive_eps=1e-6,
                         b_adaptive_max=1e7,
                         b_adaptive_disc_weight=1.0,
                         **kwargs):
    # [NEW - da_adapter] 경계 어댑터는 head/conv1/conv2 의 사전학습 재사용이 구조적으로 불가(채널 수 상이)
    #   → freshinit 의미(강제 pop + 카피 스킵) 자동 함의. (b′)의 디코더 base conv 이식은 shape 일치
    #   (384,16,3³)라 load_state_dict 가 자동 수행하므로 pop 대상이 아님.
    if da_adapter:
        zmain_fresh_init = True
    assert not (zmain_fresh_init and zmain_hybrid_init), '[init] fresh 와 hybrid 는 동시 지정 불가'
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
        dec_stages_before_conv1=dec_stages_before_conv1,  # [NEW - R3a미러]
        decoder_mirror=decoder_mirror,        # [NEW - R2 통제]
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
        # [NEW 2026-09-10 - single32] single-branch 는 enc_out_dim 이 무조건 z_dim*2 (:1309) 라
        #   z_dim != prior_z_dim 이면 head[-1] 이 사전학습(prior_z_dim*2)과 크기가 어긋난다.
        #   pop 하지 않으면 load_state_dict(strict=False) 가 **size mismatch 로 RuntimeError** 를 던진다.
        #   dual_branch=True 일 때 조건은 예전과 완전히 동일 = 기존 계보 무영향.
        if expand_encoder_head or not dual_branch:
            # encoder.head[-1] output expands from prior_z_dim*2 to z_dim*2 → shape mismatch
            _zdim_keys += ['encoder.head.2.weight', 'encoder.head.2.bias']
        for k in _zdim_keys:
            if k in state and state[k].shape != model.state_dict().get(k, state[k]).shape:
                popped[k] = state.pop(k)

        # [freshinit] residual 사슬(head.2/conv1/conv2)을 사전학습에서 분리.
        #   z16(z_dim==prior_z_dim)은 shape 일치라 위 mismatch-pop에 안 걸리고 load_state_dict가
        #   통째로 실어오므로 무조건 pop. pop된 키는 아래 카피 블록들이 fresh 분기에서 스킵되어
        #   모듈 생성 시 기본 랜덤(kaiming_uniform a=√5)이 그대로 남는다.
        #   conv2 예외: z_dim==prior_z_dim이면 conv2가 base(z_prior) 읽기 사슬과 공유(1720경
        #   conv2_prior 미생성)라 랜덤화 시 base 복원이 오염됨 → 사전학습 유지.
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
                # [freshinit] fresh면 타일/격자 카피 전체 스킵 → head[-1]·conv1은 생성 시 기본 랜덤 유지
                if expand_encoder_head and not zmain_fresh_init and not zmain_hybrid_init:
                    # [2026-08-01 hybrid init — z48 scratch] 새 채널 zero → "사전학습 행 타일(카피)"로 교체.
                    #   zero 는 새 채널이 콘텐츠-무상관 노이즈만 내보내 디코더 게이트 gradient 기대값 0
                    #   (z32 시절 52k 스텝 무이득 늪). 타일이면 복제 채널이 스텝 0부터 진짜 콘텐츠 방출.
                    #   z_dim % prior_z_dim != 0 이면 기존 zero 동작 유지. z_dim==prior_z_dim 이면 타일=1 = 기존과 동일.
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
                    # conv1 (z_dim*2→z_dim*2): zero → (mu,logvar)×군별 2×2 블록 타일.
                    #   사전학습 conv1(2h×2h)의 W_mm/W_ml/W_lm/W_ll 을 각 복제군 대각에 배치, 군간 0.
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
                # [freshinit] z16 경로의 conv1 카피도 스킵 (fresh pop으로 popped에 들어와 있어도 랜덤 유지)
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
                # [freshinit] fresh면 conv2 카피 스킵 → 랜덤 유지 (z48 전용. z16은 공유 가드로 pop 자체를 안 해
                # load_state_dict가 사전학습을 실었으므로 이 블록에 들어오지 않음)
                if 'conv2.weight' in popped and popped['conv2.weight'] is not None and not zmain_fresh_init:
                    if expand_conv2:
                        # [2026-08-01 hybrid init] zero → 사전학습 conv2(h×h) 대각 블록 타일.
                        #   전체 zero 는 디코더 게이트 zero 와 겹쳐 이중-zero 체인 (z32 늪) — 신호를 게이트까지 배달.
                        #   z_dim % prior_z_dim != 0 이면 기존 zero 유지. 함수보존은 decoder.conv1 z_main 열 zero 가 담당.
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
                        # [freshinit -r] residual 열은 모듈 생성 시 kaiming 랜덤 유지,
                        # base 열만 아래에서 사전학습으로 덮어씀
                        new_dc1w = model.decoder.conv1.weight.detach().clone()
                    else:
                        new_dc1w = torch.zeros_like(model.decoder.conv1.weight)
                    if expand_conv2:
                        # z_main ch [0:z_dim]: stays zero(or fresh random); z_prior ch [z_dim:]: pretrained
                        new_dc1w[:, z_dim:, ...] = pre_dc1w
                    else:
                        if decoder_conv1_zmain_init == 'pretrained' and not zmain_fresh_init:
                            new_dc1w[:, :prior_z_dim, ...] = pre_dc1w   # z_main ch: pretrained copy
                        # else: z_main ch stays zero (or fresh random — fresh_gate_init이 단독 결정)
                        new_dc1w[:, prior_z_dim:, ...] = pre_dc1w       # z_prior ch: pretrained copy
                    model.decoder.conv1.weight.copy_(new_dc1w)
                    if 'decoder.conv1.bias' in popped and popped['decoder.conv1.bias'] is not None:
                        model.decoder.conv1.bias.copy_(popped['decoder.conv1.bias'])

                # [NEW - hybrid] z_main 첫 prior_z_dim 채널 = 사전학습 z16 경로를 정확히 재현,
                #   나머지 신규 채널 = 생성 시 kaiming 랜덤 유지 (타일 복제도 zero 도 아님).
                #   근거: z16 실험에서 1:1 승계가 랜덤보다 우세(−0.14dB) + z48 타일은 복사군 corr 0.949 로 미분화.
                if zmain_hybrid_init and expand_encoder_head:
                    _h = prior_z_dim
                    _peh = popped.get('encoder.head.2.weight')
                    if _peh is not None:
                        model.encoder.head[-1].weight[:_h].copy_(_peh[:_h])                    # mu 첫 군
                        model.encoder.head[-1].weight[z_dim:z_dim + _h].copy_(_peh[_h:])       # logvar 첫 군
                    _pebv = popped.get('encoder.head.2.bias')
                    if _pebv is not None:
                        model.encoder.head[-1].bias[:_h].copy_(_pebv[:_h])
                        model.encoder.head[-1].bias[z_dim:z_dim + _h].copy_(_pebv[_h:])
                    # conv1: 첫 군의 (mu,logvar) 2×2 블록만 사전학습, 첫 군 행의 나머지 열은 0 (타 군 오염 차단)
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
                    # conv2: 첫 군 행만 사전학습(첫 군 열), 나머지 열 0. 타 군 행은 랜덤 유지
                    _pc2 = popped.get('conv2.weight')
                    if _pc2 is not None and expand_conv2 and _pc2.shape[0] == _h:
                        model.conv2.weight[:_h].zero_()
                        model.conv2.weight[:_h, :_h] = _pc2
                        _pc2b = popped.get('conv2.bias')
                        if model.conv2.bias is not None and _pc2b is not None:
                            model.conv2.bias[:_h] = _pc2b
                    logging.info(f'[hybrid] z_main 첫 {_h}채널 = 사전학습 승계, 나머지 {z_dim - _h}채널 = 랜덤')

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

            # prior_encoder: pretrained encoder 가중치 복사 후 freeze
            prior_state = {
                k[len('encoder.'):]: v
                for k, v in state.items()
                if k.startswith('encoder.') and not k.startswith('encoder.add_downsamples')
            }
            # expand_encoder_head=True 시 encoder.head.2.weight/bias가 popped → prior_state에서 누락됨
            # prior_encoder.head[-1]은 항상 prior_z_dim*2 출력 → pretrained 원본 shape과 동일하므로 popped에서 복원
            # [freshinit] fresh pop(z16 포함)도 같은 누락을 만들므로 동일하게 복원
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
        # [NEW 2026-09-10 - single32] composite 없는 단일 인코더 전용 초기화.
        #   위 dual 블록을 재사용하지 않고 따로 짠다. 이유:
        #     dual 의 카피 코드는 "z_prior 16채널이 동결 Wan 이라 잠재의 절반이 sigma≈0 으로
        #     고정된다" 는 전제로 쓰였다. single 엔 그 앵커가 없어서 신규 채널의 logvar 가
        #     통제되지 않는다 — 실측: 신규 16채널의 logvar 평균이 -1~+4 (승계 채널은 -65),
        #     그중 5개가 양수 → sigma 1~5 → 학습 시 latents_std 가 6~615 로 요동하고
        #     rec 이 안 내려간다(같은 조건 composite 대조군은 2.2 로 안정).
        #
        #   설계 (첫 _h = prior_z_dim 채널은 사전학습 z16 경로를 그대로 재현):
        #     [A] head[-1]  mu/logvar 첫 군만 사전학습. 신규 군의 mu·logvar 행은 랜덤 유지
        #                   — 여기를 0 으로 두면 아래 conv1 의 0 과 직렬이 되어 기울기가
        #                     영영 안 흐른다. zero-gate 는 반드시 한쪽만.
        #     [B] conv1     첫 군 행 = 사전학습 2x2 블록(타 군 열 0). 신규 mu 행은 랜덤 유지,
        #                   **신규 logvar 행 = zero weight + bias _LV0** ← 이번 진동의 fix.
        #                   sigma 가 상수 e^(_LV0/2) 에서 출발하고 bias/weight 로 학습된다.
        #     [C] conv2     첫 군 행의 첫 군 열만 사전학습. 신규 행은 랜덤 유지
        #                   (0 으로 두면 decoder 신규 열이 기울기를 못 받아 영영 안 열린다).
        #     [D] dec.conv1 첫 군 열 = 사전학습, 신규 열 = 0 (no-op 출발).
        #                   ★ dual 처럼 첫 열을 0 으로 두면 디코더 입력이 통째로 0 = dead start.
        # ═══════════════════════════════════════════════════════════════════
        elif z_dim != prior_z_dim:
            _h = prior_z_dim
            _LV0 = -10.0        # sigma = e^(-5) ~= 0.0067 (승계 채널의 실측 logvar 는 -65 근처)
            with torch.no_grad():
                # [A] head[-1]
                _ph, _pb = popped.get('encoder.head.2.weight'), popped.get('encoder.head.2.bias')
                if _ph is not None:
                    model.encoder.head[-1].weight[:_h].copy_(_ph[:_h])                 # mu 첫 군
                    model.encoder.head[-1].weight[z_dim:z_dim + _h].copy_(_ph[_h:])    # logvar 첫 군
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
                model.conv1.weight[z_dim + _h:].zero_()               # 신규 logvar 행
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
            logging.info(f'[single32] 첫 {_h}채널 = 사전학습 z{_h} 경로 재현 / '
                         f'신규 {z_dim - _h}채널 = mu 랜덤 + logvar {_LV0} 출발 / '
                         f'decoder 신규 열 = 0 (no-op 출발)')

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
