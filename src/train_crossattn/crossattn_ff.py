"""
[crossattn] 첫 프레임(keyframe) cross-attention injection 모듈.

Reducio-VAE (ldm/models/reducio_vae.py) 에서 이식:
  - window_partition / window_reverse (line 144-177): 고해상도 attention 비용을 window 로 선형화.
  - CrossAttention (line 268-341): windowed cross-attn. query=decoder feature(5D b,d,T,H,W),
    context=keyframe feature(4D b,d,H',W'). window_size 스케일링으로 query/context 해상도 불일치 흡수.
  - GatedCrossAttnBlock (= Reducio BasicTransformerBlock line 344-419): LayerScale gate(gamma).
    gamma=0 init → out = x (기존 decode 와 bit-identical, finetune 보존). t2_projection zero-init 과 동일 철학.

attention 백엔드: diffsynth(stage2) wan_video_dit.py 의 flash_attention 디스패처 이식
  → FA4(B200 native, CuTeDSL) → FA3 → FA2 → sage → SDPA 자동선택. cross-attn(q_seq≠kv_seq) 지원.
  flash 는 bf16/fp16 필요 → autocast 아래서 사용, fp32 면 SDPA 폴백.

keyframe feature 소스: [변경 2026-07] 자작 FFPyramidEncoder(from-scratch) 폐기 →
  첫 프레임을 frozen VAE encoder 에 통과시켜 각 다운샘플 "전" 중간 feature(return_skip) 를 K/V 로 사용.
  사전학습 feature (첫프레임 detail 39~44dB 보존 실측), 새 param 0.
  4-레벨 skip 채널 [skip@256=96, skip@128=192, skip@64=384, skip@32=384],
  decoder 주입 레벨(coarse→fine) [ups2=384@32, ups6=384@64, ups10=192@128, ups14=96@256] 와 1:1 대칭.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --- attention 백엔드 디스패처 (diffsynth wan_video_dit.py:11-79 이식) ---
try:
    from flash_attn.cute import flash_attn_func as _v4_flash_attn_func
    FLASH_ATTN_4_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_4_AVAILABLE = False
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def flash_attention(q, k, v, num_heads):
    """q,k,v: (B, S, num_heads*head_dim). cross-attn 시 q_S != kv_S 허용. return (B, Sq, num_heads*head_dim)."""
    # flash 계열은 bf16/fp16 만 지원 → fp32 면 SDPA 폴백.
    flash_ok = (FLASH_ATTN_4_AVAILABLE or FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_2_AVAILABLE) \
        and q.dtype in (torch.float16, torch.bfloat16)
    if flash_ok and FLASH_ATTN_4_AVAILABLE:
        q, k, v = (rearrange(t, "b s (n d) -> b s n d", n=num_heads) for t in (q, k, v))
        x = _v4_flash_attn_func(q, k, v)
        x = x[0] if isinstance(x, tuple) else x
        return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif flash_ok and FLASH_ATTN_3_AVAILABLE:
        q, k, v = (rearrange(t, "b s (n d) -> b s n d", n=num_heads) for t in (q, k, v))
        x = flash_attn_interface.flash_attn_func(q, k, v)
        x = x[0] if isinstance(x, tuple) else x
        return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif flash_ok and FLASH_ATTN_2_AVAILABLE:
        q, k, v = (rearrange(t, "b s (n d) -> b s n d", n=num_heads) for t in (q, k, v))
        x = flash_attn.flash_attn_func(q, k, v)
        return rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:  # SDPA 폴백 (fp32 안전, 기존 VAE AttentionBlock 과 동일 백엔드)
        q, k, v = (rearrange(t, "b s (n d) -> b n s d", n=num_heads) for t in (q, k, v))
        x = F.scaled_dot_product_attention(q, k, v)
        return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def default(val, d):
    return val if val is not None else d


# ---------------------------------------------------------------------------
# window partition / reverse  (Reducio reducio_vae.py:144-177)
# ---------------------------------------------------------------------------
def window_partition(x, window_size):
    """x: (B,C,H,W)[4D] 또는 (B,C,T,H,W)[5D]. return (num_win*B, seq, C)."""
    if x.ndim == 4:
        return rearrange(x, "b c (h w1) (w w2) -> (b h w) (w1 w2) c",
                         w1=window_size[0], w2=window_size[1])
    return rearrange(x, "b c t (h w1) (w w2) -> (b h w) (t w1 w2) c",
                     w1=window_size[0], w2=window_size[1])


def window_reverse(x, window_size, T, H, W):
    """x: (num_win*B, T*w1*w2, C) → (B,C,T,H,W). H,W = padded 크기."""
    return rearrange(x, "(b h w) (t w1 w2) c -> b c t (h w1) (w w2)",
                     t=T, h=H // window_size[0], w=W // window_size[1],
                     w1=window_size[0], w2=window_size[1])


# ---------------------------------------------------------------------------
# CrossAttention  (Reducio reducio_vae.py:268-341, backend=flash_attention 디스패처)
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64,
                 window_size=(32, 32), qkv_bias=False, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.heads = heads
        self.window_size = window_size
        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=qkv_bias)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context):
        # x: (b,d,T,H,W)  context: (b,d,H_,W_)
        b, _, T, H, W = x.shape
        H_, W_ = context.shape[-2], context.shape[-1]
        ws0, ws1 = self.window_size
        # 저해상도(map<=window)는 flat 경로로 자동 폴백 (partition 에러 회피).
        enable_window = (H_ * W_ > ws0 * ws1) and (H * W >= ws0 * ws1)

        if enable_window:
            pad_r2 = (ws1 - W_ % ws1) % ws1
            pad_b2 = (ws0 - H_ % ws0) % ws0
            context = F.pad(context, (0, pad_r2, 0, pad_b2))
            ws_q = (math.ceil(ws0 * H / H_), math.ceil(ws1 * W / W_))  # query window 스케일(Reducio line 301)
            pad_r3 = (ws_q[1] - W % ws_q[1]) % ws_q[1]
            pad_b3 = (ws_q[0] - H % ws_q[0]) % ws_q[0]
            x = F.pad(x, (0, pad_r3, 0, pad_b3))
            _, _, _, Hp, Wp = x.shape
            ctx = window_partition(context, self.window_size)   # (nw*b, w1*w2, d)
            qry = window_partition(x, ws_q)                     # (nw*b, T*wq1*wq2, d)
        else:
            qry = rearrange(x, 'b d t h w -> b (t h w) d')
            ctx = rearrange(context, 'b d h w -> b (h w) d')

        q = self.to_q(qry)                                      # (B_eff, Sq, inner)
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        out = flash_attention(q, k, v, self.heads)              # (B_eff, Sq, inner)

        if enable_window:
            out = window_reverse(out, ws_q, T, Hp, Wp)          # (b,d,T,Hp,Wp)
            if pad_r3 or pad_b3:
                out = out[:, :, :, :H, :W].contiguous()
            out = rearrange(out, 'b c t h w -> b (t h w) c')
        return self.to_out(out)                                 # (b, T*H*W, query_dim)


# ---------------------------------------------------------------------------
# GatedCrossAttnBlock  (Reducio BasicTransformerBlock line 344-419)
#   out = x + gamma * cross_attn(norm(x), context),  gamma=0 init → identity(finetune 보존)
# ---------------------------------------------------------------------------
class GatedCrossAttnBlock(nn.Module):
    def __init__(self, dim, context_dim, d_head=32, window_size=(32, 32), init_values=0.0):
        super().__init__()
        n_heads = max(1, dim // d_head)
        self.norm = nn.LayerNorm(dim)
        self.attn = CrossAttention(query_dim=dim, context_dim=context_dim,
                                   heads=n_heads, dim_head=d_head, window_size=window_size)
        self.gamma = nn.Parameter(init_values * torch.ones(dim))  # LayerScale gate, init 0 → no-op

    def forward(self, x, context):
        # x: (b,d,t,h,w)  context: (b,d,h',w')
        b, d, t, hh, ww = x.shape
        xn = rearrange(x, 'b d t h w -> b (t h w) d')
        xn = self.norm(xn)
        xn = rearrange(xn, 'b (t h w) d -> b d t h w', t=t, h=hh, w=ww)
        out = self.attn(xn, context)                            # (b, t*h*w, d)
        out = rearrange(out, 'b (t h w) d -> b d t h w', t=t, h=hh, w=ww)
        return x + self.gamma.view(1, d, 1, 1, 1) * out

    @torch.no_grad()
    def gamma_abs_mean(self):
        return self.gamma.abs().mean().item()

# FFPyramidEncoder 삭제 (2026-07): keyframe feature 는 frozen VAE encoder 의 return_skip 중간 feature 로 대체.
