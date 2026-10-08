"""[NEW - oliviaa] AdaptiveWeightedCausalConv3d — (B) 식 의 정확 한 구현.

기존 single backward path 의 _AdaptiveWeightingFn (= z view 식, activation gradient ratio) 의
대안. (B) 식 의 정확 한 weight gradient ratio 측정 + scale 적용.

특징:
  - forward = F.conv3d 직접 + return (y_main, y_adversarial) (= same value, separate backward edges)
  - backward = manual conv weight gradient (= torch.nn.grad.conv3d_weight) + DDP all_reduce + adaptive ratio + scale
  - scope = encoder body 까지 (= x 와 W 의 grad 둘 다 scale 적용)
  - DDP / FSDP 호환 (= all_reduce 으로 sync)

VQGAN-style adaptive weighting 의 응용 — generator 의 학습 균형 의 의도.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _zero_like_branch_grad(*branch_grads):
    for branch_grad in branch_grads:
        if branch_grad is not None:
            return torch.zeros_like(branch_grad)
    raise RuntimeError("At least one branch gradient must be non-None.")


class _AdaptiveWeightedConv3dFn(torch.autograd.Function):
    """Custom autograd Function — (B) 식.

    forward:
      input = (x, weight, bias, stride, padding, dilation, groups, ...)
      output = (y, y)  ← same forward value, separate backward edges

    backward:
      - manual conv weight gradient (= torch.nn.grad.conv3d_weight) × 2
      - DDP all_reduce 으로 norm sync (= 모든 rank 가 같은 ratio)
      - adaptive_weight = ‖grad_W_main‖ / ‖grad_W_adv‖ × disc_weight
      - grad_x, grad_weight 의 align contribution 에 scale 적용

    Mixed precision (= autocast bf16):
      @custom_fwd / @custom_bwd decorator 으로 forward / backward 의 dtype 자동 일치.
      (autocast 의 default = forward 시 input bf16 cast → ctx.save 의 dtype 가 bf16 → backward 시 grad_output bf16 와 일치)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        adaptive_weight_eps,
        adaptive_weight_max,
        disc_weight,
    ):
        # [FIX v2 - oliviaa/B-fix] 2backward 와 dtype 정확 일치 위해:
        #   ctx.save = 원본 dtype (= float32) 유지 (= autograd 의 자동 cast emulation)
        #   forward 의 F.conv3d 는 cast 된 dtype (= bf16) — autocast 의 안 의 동작 모방
        # 이렇게 하면 backward 의 grad_W 계산 = float32 (= 2backward 와 동일)
        if torch.is_autocast_enabled():
            _ac_dtype = torch.get_autocast_gpu_dtype()
            _x_fwd = x.to(_ac_dtype) if x.dtype != _ac_dtype else x
            _w_fwd = weight.to(_ac_dtype) if weight.dtype != _ac_dtype else weight
            _b_fwd = bias.to(_ac_dtype) if (bias is not None and bias.dtype != _ac_dtype) else bias
        else:
            _x_fwd, _w_fwd, _b_fwd = x, weight, bias

        # ctx.save 는 원본 dtype (= backward 의 자동 cast 의 emulation 위해)
        ctx.save_for_backward(x, weight, bias)
        ctx.meta = (
            stride,
            padding,
            dilation,
            groups,
            adaptive_weight_eps,
            adaptive_weight_max,
            disc_weight,
        )

        # forward = cast 된 dtype (= autocast 의 자동 cast 와 동일)
        y = F.conv3d(
            _x_fwd,
            _w_fwd,
            _b_fwd,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        # Same forward value, separate backward edges.
        # [FIX - oliviaa] return y, y 시 autograd 가 same tensor → single output 처리 가능 →
        # backward 시 separate grad 받기 위해 별도 tensor (= y.clone()) 으로 분리
        # Intended use:
        #   y_main -> reconstruction / perceptual / NLL loss
        #   y_adv  -> generator adversarial / align loss
        return y, y.clone()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_y_main, grad_y_adversarial):
        x, weight, bias = ctx.saved_tensors
        (
            stride,
            padding,
            dilation,
            groups,
            adaptive_weight_eps,
            adaptive_weight_max,
            disc_weight,
        ) = ctx.meta

        # [FIX - oliviaa/B-fix verify] verify-style call (= autograd.grad(single_loss, W)) 시
        # 한 branch 만 non-None. 기존 _zero_like_branch_grad 가 None 자리 0 으로 채워 ratio = 0/x = 0
        # → grad_W = 0 (= bug). 해결: 한 branch 만 들어오면 plain conv backward, adaptive weighting skip.
        if grad_y_main is None and grad_y_adversarial is None:
            return (None,) * 10
        if grad_y_main is None or grad_y_adversarial is None:
            _gy = grad_y_adversarial if grad_y_main is None else grad_y_main
            _target_dtype = weight.dtype
            if _gy.dtype != _target_dtype:
                _gy = _gy.to(_target_dtype)
            _gW = torch.nn.grad.conv3d_weight(input=x, weight_size=weight.shape, grad_output=_gy,
                                              stride=stride, padding=padding, dilation=dilation, groups=groups)
            _gx = torch.nn.grad.conv3d_input(input_size=x.shape, weight=weight, grad_output=_gy,
                                             stride=stride, padding=padding, dilation=dilation, groups=groups)
            _gb = _gy.sum(dim=(0, 2, 3, 4)) if bias is not None else None
            return (_gx, _gW, _gb, None, None, None, None, None, None, None)

        # [FIX v2 - oliviaa/B-fix] 2backward 와 dtype 정확 일치 위해:
        #   - ctx.saved x, weight = 원본 dtype (= float32)
        #   - grad_y = bf16 (= autocast forward 의 결과)
        #   - grad_y 를 weight 의 dtype (= float32) 으로 cast → 모든 backward 계산 float32
        #   - autograd 의 자동 cast 의 정확 emulation
        _target_dtype = weight.dtype  # = 원본 (= float32, 2backward 와 동일)
        if grad_y_main.dtype != _target_dtype:
            grad_y_main = grad_y_main.to(_target_dtype)
        if grad_y_adversarial.dtype != _target_dtype:
            grad_y_adversarial = grad_y_adversarial.to(_target_dtype)

        # ─── manual conv weight gradient × 2 (= sum 전 별도) ──────────
        grad_weight_main = torch.nn.grad.conv3d_weight(
            input=x,
            weight_size=weight.shape,
            grad_output=grad_y_main,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        grad_weight_adversarial = torch.nn.grad.conv3d_weight(
            input=x,
            weight_size=weight.shape,
            grad_output=grad_y_adversarial,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        # ─── DDP all_reduce + adaptive ratio (= 2backward 와 정확 일치) ───────
        # [FIX v4 - oliviaa/B-fix] eps 의 mathematical 처리 = 2backward 와 정확 일치
        # 2backward: ‖a‖ / (‖b‖ + eps)
        # (B):       ‖a‖ / (‖b‖ + eps)  ← 수정 (= 이전 sqrt(‖a‖²/(‖b‖²+eps)) 와 다른 함수)
        # all_reduce 의 의 norm² 의 sum (= global norm² = sum_per_rank(per_rank_norm²)) → sqrt
        n_main = torch.linalg.vector_norm(grad_weight_main).pow(2)
        n_adv  = torch.linalg.vector_norm(grad_weight_adversarial).pow(2)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(n_main, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_adv,  op=dist.ReduceOp.SUM)

        # global norm = sqrt(sum of per-rank norm²)
        global_norm_main = n_main.sqrt()
        global_norm_adv  = n_adv.sqrt()
        adaptive_weight_raw = global_norm_main / (global_norm_adv + adaptive_weight_eps)
        adaptive_weight = torch.clamp(adaptive_weight_raw, 0.0, adaptive_weight_max)
        adaptive_weight = adaptive_weight.detach() * disc_weight

        # _last_c 저장 (= logging 위해 — _AdaptiveWeightingFn 와 호환)
        _AdaptiveWeightedConv3dFn._last_c = adaptive_weight.detach()
        _AdaptiveWeightedConv3dFn._last_c_raw = adaptive_weight_raw.detach()
        # [DEBUG v23] (B) backward 의 intermediate quantity log (= train.py 의 2bwd 측정 과 비교)
        _AdaptiveWeightedConv3dFn._last_grad_y_main_norm = grad_y_main.detach().float().norm()
        _AdaptiveWeightedConv3dFn._last_grad_y_adv_norm = grad_y_adversarial.detach().float().norm()
        _AdaptiveWeightedConv3dFn._last_grad_W_main_norm = global_norm_main.detach()
        _AdaptiveWeightedConv3dFn._last_grad_W_adv_norm = global_norm_adv.detach()

        # ─── scale 적용 한 grad_y → grad_x, grad_weight 계산 ─────────
        grad_y = grad_y_main + adaptive_weight * grad_y_adversarial

        grad_x = torch.nn.grad.conv3d_input(
            input_size=x.shape,
            weight=weight,
            grad_output=grad_y,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        grad_weight = grad_weight_main + adaptive_weight * grad_weight_adversarial

        grad_bias = None
        if bias is not None:
            grad_bias = grad_y.sum(dim=(0, 2, 3, 4))

        return (
            grad_x,
            grad_weight,
            grad_bias,
            None,  # stride
            None,  # padding
            None,  # dilation
            None,  # groups
            None,  # adaptive_weight_eps
            None,  # adaptive_weight_max
            None,  # disc_weight
        )


class AdaptiveWeightedConv3d(nn.Module):
    """Conv3d with duplicated outputs and VQGAN-style adaptive adversarial weighting.

    Forward returns: (y_main, y_adversarial)
      - same forward value, separate backward edges.

    Backward behaves like:
      grad = grad_main + (disc_weight * adaptive_weight) * grad_adversarial
      adaptive_weight = ‖∇_W grad_main‖ / (‖∇_W grad_adv‖ + eps)
      (모든 rank 에서 sync — DDP all_reduce 으로)
    """

    def __init__(
        self,
        adaptive_weight_eps=1e-6,
        adaptive_weight_max=1e7,
        disc_weight=1.0,
    ):
        super().__init__()
        self.adaptive_weight_eps = adaptive_weight_eps
        self.adaptive_weight_max = adaptive_weight_max
        self.disc_weight = disc_weight

    def forward(
        self,
        x,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        return _AdaptiveWeightedConv3dFn.apply(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            self.adaptive_weight_eps,
            self.adaptive_weight_max,
            self.disc_weight,
        )


class AdaptiveWeightedCausalConv3d(nn.Conv3d):
    """Causal 3D convolution with VQGAN-style adaptive adversarial weighting.

    By default, this module returns two identical forward outputs:
        y_main, y_adversarial
    The two outputs have separate backward edges.

    During backward, the adversarial branch is rescaled by an adaptive weight
    computed from the ratio of the convolution weight-gradient norms.
    DDP / FSDP 환경 에서 all_reduce 으로 모든 rank 가 같은 ratio 사용.

    Use return_duplicate_y=False for normal single-output behavior.
    """

    def __init__(
        self,
        *args,
        adaptive_weight_eps=1e-6,
        adaptive_weight_max=1e7,
        disc_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # CausalConv3d 의 비대칭 padding 처리
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

        self.adaptive_conv3d = AdaptiveWeightedConv3d(
            adaptive_weight_eps=adaptive_weight_eps,
            adaptive_weight_max=adaptive_weight_max,
            disc_weight=disc_weight,
        )

    def _apply_causal_padding(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(device=x.device, dtype=x.dtype)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        return F.pad(x, padding)

    def forward(self, x, cache_x=None, return_duplicate_y=True):
        x = self._apply_causal_padding(x, cache_x=cache_x)

        y_main, y_adversarial = self.adaptive_conv3d(
            x=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=0,  # causal padding has already been applied by F.pad
            dilation=self.dilation,
            groups=self.groups,
        )

        if return_duplicate_y:
            return y_main, y_adversarial
        return y_main
