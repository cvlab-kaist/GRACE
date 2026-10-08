"""Dual-schedule core: per-branch Wan flow-match sigma ladders + Euler step.
Copied from the dual_schedule repo (hyunbin); sys.path adjusted for this inference repo.
"""
import os, sys
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_DIFFSYNTH = os.environ.get("DIFFSYNTH_ROOT", os.path.join(os.path.dirname(_ROOT), "DiffSynth-Studio"))
for p in [_DIFFSYNTH]:
    if p and p not in sys.path:
        sys.path.insert(0, p)
import torch
from diffsynth.diffusion import FlowMatchScheduler


def wan_sigmas(num_steps, shift):
    """The exact Wan flow-match sigma ladder for a given shift (β)."""
    s = FlowMatchScheduler("Wan")
    s.set_timesteps(num_steps, denoising_strength=1.0, shift=shift)
    return s.sigmas


def async_ladders(num_steps, shift, delta, sigma_min=0.02, shift_prior=None, sigma_max=1.0, prior_tail_knee=None):
    """[async] Pair of offset ladders, the same geometry the async loss was trained with.

    master m: (u_top+delta) -> 0 on the same grid convention as wan_sigmas (linspace over N+1, final
      point dropped), so the number of function evaluations is fixed and only the spacing widens.
    delta: a scalar, or an array of shape (num_steps,) for a custom path inside the wedge.
    phase 1 (m > u_top): u_main is held at u_top, so sigma is constant and dual_step takes d=0 (main waits).
    phase 3 (m - delta < 0): the sigma_min floor applies only in the saturated region, while the base
      finishes - the same treatment as in training.
    With delta=0 both ladders equal wan_sigmas(num_steps, shift) exactly, which guarantees no regression.
    sigma_max: the master clock's starting sigma, for truncated start / FrameInit. 1.0 is the old
      behaviour (u_top=1). With sigma0 < 1 the bound is cut to u_top = warp^-1(sigma0), so the ladder
      starts at sigma0 and the declared noise level matches the actual signal content (1-sigma0),
      which fixes the brightness blowout.
    """
    d = torch.as_tensor(delta, dtype=torch.float32)
    d = d.expand(num_steps).clone() if d.ndim == 0 else d
    assert d.shape == (num_steps,), f"delta must be a scalar or an array of shape ({num_steps},): {tuple(d.shape)}"
    dmax = float(d.max().clamp(min=0.0))
    # u_top = warp^-1(sigma_max, shift): the u that corresponds to sigma0. sigma_max=1.0 gives u_top=1.0.
    u_top = float(sigma_max) / (shift - (shift - 1.0) * float(sigma_max))
    m = torch.linspace(u_top + dmax, 0.0, num_steps + 1)[:-1]   # the wan_sigmas grid convention (the final 0 is excluded)
    u_main = m.clamp(0.0, u_top)
    u_prior = (m - d).clamp(0.0, u_top)
    # separate shifts for main and prior. With shift_prior=None both use shift, which is the old behaviour.
    #   The offset (delta) is kept and only the base is warped with shift_prior - offset plus asymmetric warp.
    warp = lambda u, sh: sh * u / (1.0 + (sh - 1.0) * u)      # same expression as wan_sigmas
    _sp = shift if shift_prior is None else float(shift_prior)
    sig_main = warp(u_main, shift)
    sig_prior = warp(u_prior, _sp)
    sig_prior = torch.where(m - d < 0.0, sig_prior.clamp(min=sigma_min), sig_prior)
    # [tail descent] Instead of freezing the base after it commits (constant sigma, zero step), it descends
    #   gently from the knee to sigma_min over all remaining steps, so the base keeps being corrected
    #   against main's current state. Monotonicity is guaranteed by clamping the part that would fall
    #   below the knee first, then attaching the tail, which prevents going back up.
    if prior_tail_knee is not None and prior_tail_knee > sigma_min:
        kv = float(prior_tail_knee)
        below = (sig_prior <= kv).nonzero()
        if len(below) > 0:
            k = int(below[0])
            if k < num_steps - 1:
                sig_prior = sig_prior.clone()
                sig_prior[k:] = torch.linspace(kv, sigma_min, num_steps - k)
    return sig_main, sig_prior


def async_ladders_decoupled(num_steps, shift, offset):
    """Decoupled ladders: the base runs the full [1,0] with nothing wasted, while main runs
    [1+offset,0] and starts frozen. The gap is offset*u, so it shrinks from offset to 0, and both
    reach clean. The lead is weaker than with a constant offset, closer to synchronous.
    """
    warp = lambda u: shift * u / (1.0 + (shift - 1.0) * u)
    up = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]                            # prior: full [1,0]
    um = torch.linspace(1.0 + offset, 0.0, num_steps + 1)[:-1].clamp(0.0, 1.0)   # main: [1+offset,0] clamp
    return warp(um), warp(up)


def dual_step(latents, velocity, i, sig_main, sig_prior, z_dim):
    """One Euler step where z_main and z_prior advance on their own sigma ladders.
    latents/velocity: (B, Z+P, T, H, W). sig_*: 1-D ladders (CPU float32 from wan_sigmas;
    caller moves them to device if needed). The terminal step (i+1 past the ladder end) uses
    sigma_next = 0, matching DiffSynth FlowMatchScheduler.step(to_final=True); note the last
    ladder entry is NOT ~0, so the final delta is the full residual."""
    nm, np_ = len(sig_main), len(sig_prior)
    nxt_m = sig_main[i + 1] if i + 1 < nm else sig_main.new_zeros(())
    nxt_p = sig_prior[i + 1] if i + 1 < np_ else sig_prior.new_zeros(())
    d_main = (nxt_m - sig_main[i]).item()
    d_prior = (nxt_p - sig_prior[i]).item()
    z_main = latents[:, :z_dim] + velocity[:, :z_dim] * d_main
    z_prior = latents[:, z_dim:] + velocity[:, z_dim:] * d_prior
    return torch.cat([z_main, z_prior], dim=1)
