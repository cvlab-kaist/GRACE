"""AdaptiveWeightedCausalConv3d - the exact form of the (B) weighting.

An alternative to the single-backward-path _AdaptiveWeightingFn, which measures an activation
gradient ratio in z space. This one measures the weight gradient ratio exactly and applies the
resulting scale.

  - forward: F.conv3d directly, returning (y_main, y_adversarial) - the same value on two
    separate backward edges.
  - backward: a manual conv weight gradient (torch.nn.grad.conv3d_weight), a DDP all_reduce,
    then the adaptive ratio and the scale.
  - scope: through the encoder body, so the scale reaches the gradients of both x and W.
  - works under DDP and FSDP, since ranks synchronise through all_reduce.

It is VQGAN-style adaptive weighting, used here to balance what the generator learns.
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
    """Custom autograd Function for the (B) weighting.

    forward:
      input = (x, weight, bias, stride, padding, dilation, groups, ...)
      output = (y, y)  ← same forward value, separate backward edges

    backward:
      - manual conv weight gradient (= torch.nn.grad.conv3d_weight) × 2
      - norms synced through a DDP all_reduce, so every rank uses the same ratio
      - adaptive_weight = ‖grad_W_main‖ / ‖grad_W_adv‖ × disc_weight
      - the scale is applied to the align contribution of grad_x and grad_weight

    Mixed precision (= autocast bf16):
      The @custom_fwd / @custom_bwd decorators keep the forward and backward dtypes in step:
      autocast casts the input to bf16 on the way in, so what ctx saves is bf16 and matches the
      bf16 grad_output on the way back.
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
        # [FIX v2] to match the two-backward path's dtypes exactly:
        #   what ctx saves keeps the original dtype (float32), emulating autograd's own cast;
        #   the F.conv3d in forward runs on the cast dtype (bf16), as it would inside autocast.
        # grad_W in backward is then computed in float32, the same as two-backward.
        if torch.is_autocast_enabled():
            _ac_dtype = torch.get_autocast_gpu_dtype()
            _x_fwd = x.to(_ac_dtype) if x.dtype != _ac_dtype else x
            _w_fwd = weight.to(_ac_dtype) if weight.dtype != _ac_dtype else weight
            _b_fwd = bias.to(_ac_dtype) if (bias is not None and bias.dtype != _ac_dtype) else bias
        else:
            _x_fwd, _w_fwd, _b_fwd = x, weight, bias

        # ctx saves the original dtype, to emulate the automatic cast on the way back
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

        # forward runs on the cast dtype, as autocast would do it
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
        # [FIX] returning y twice lets autograd see one tensor and treat it as a single output,
        # so split it into a separate tensor (y.clone()) to get separate gradients back.
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

        # [FIX] a verify-style call, autograd.grad(single_loss, W), leaves only one branch non-None.
        # _zero_like_branch_grad used to fill the None with zeros, making the ratio 0/x = 0 and
        # grad_W zero. Now a single branch takes a plain conv backward and skips adaptive weighting.
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

        # [FIX v2] to match the two-backward path's dtypes exactly:
        #   - the saved x and weight are the original dtype (float32)
        #   - grad_y arrives as bf16, out of the autocast forward
        #   - cast grad_y to weight's dtype, so every backward computation runs in float32
        # which is exactly what autograd's automatic cast does.
        _target_dtype = weight.dtype  # the original float32, as in two-backward
        if grad_y_main.dtype != _target_dtype:
            grad_y_main = grad_y_main.to(_target_dtype)
        if grad_y_adversarial.dtype != _target_dtype:
            grad_y_adversarial = grad_y_adversarial.to(_target_dtype)

        # --- manual conv weight gradient, twice, kept separate before the sum ---
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

        # --- DDP all_reduce, then the adaptive ratio, matching two-backward exactly ---
        # [FIX v4] eps is handled the same way two-backward handles it
        # 2backward: ‖a‖ / (‖b‖ + eps)
        # (B):       ‖a‖ / (‖b‖ + eps)  <- the fix; the old sqrt(‖a‖²/(‖b‖²+eps)) is a different function
        # all_reduce sums the squared norms into a global norm², then takes the square root
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

        # keep _last_c for logging, the same field _AdaptiveWeightingFn uses
        _AdaptiveWeightedConv3dFn._last_c = adaptive_weight.detach()
        _AdaptiveWeightedConv3dFn._last_c_raw = adaptive_weight_raw.detach()
        # [DEBUG v23] log the intermediate quantities of the (B) backward, to compare against
        # the two-backward measurement in train.py
        _AdaptiveWeightedConv3dFn._last_grad_y_main_norm = grad_y_main.detach().float().norm()
        _AdaptiveWeightedConv3dFn._last_grad_y_adv_norm = grad_y_adversarial.detach().float().norm()
        _AdaptiveWeightedConv3dFn._last_grad_W_main_norm = global_norm_main.detach()
        _AdaptiveWeightedConv3dFn._last_grad_W_adv_norm = global_norm_adv.detach()

        # --- apply the scale to grad_y, then compute grad_x and grad_weight ---
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
      (synced across ranks through a DDP all_reduce)
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
    Under DDP or FSDP, an all_reduce gives every rank the same ratio.

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

        # CausalConv3d's asymmetric padding
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
