# What this fork changes

A line-based patch against upstream used to live here, but it had drifted out of sync with the
files it described and no longer applied. This note replaces it: the files in this directory are
the authoritative copy, and this is what they change relative to upstream DiffSynth-Studio.

Every change is marked in the source with `[NEW]`, `[Modified]`, `[FIX]` or `[batch patch]`, so
`grep -rn '\[NEW\|\[Modified\|\[FIX\|\[batch patch' diffsynth/` lists them in place.

## Inference, and why upstream gives different videos

These two are the reason the README tells you to use this copy. Upstream accepts the settings and
ignores them, so a run finishes without error and returns a video that is not what the paper reports.

- **`rope_pos_scale`** (`diffsynth/pipelines/wan_video.py`) — subsample-aware slicing of the RoPE
  axes. Our checkpoints were trained at `(2, 2, 2)`. A value of `None` or `1` is bit-identical to
  upstream, which keeps the change opt-in.
- **Second timestep `timestep2`** (same file) — the opt-in injection of `t_prior` for asymmetric
  denoising, the same block training uses. It needs both the kwarg from the caller and a
  `t2_projection` on the DiT, so every other path is untouched.

Also in inference:

- A warning when a latent's H or W is not a multiple of 2, because a stride-2 `Conv3d` silently
  drops the last row or column.
- FlashAttention 4 (`flash_attn.cute`, Hopper and Blackwell native) added to the dispatcher ahead
  of FA3, FA2 and SDPA (`diffsynth/models/wan_video_dit.py`).

## Training

Not needed to generate videos; listed so the diff is accounted for.

- **Batch patches** (`diffsynth/diffusion/flow_match.py`, `diffsynth/models/wan_video_dit.py`) —
  per-sample timesteps of shape `(B,)`. Upstream's modulation broadcast only happened to work at
  `B=2` and mismatches for `B > 2`; scalars still work.
- **Resume** (`diffsynth/diffusion/{logger,runner}.py`) — `accelerator.save_state` writes the
  optimizer, scheduler and model together, the dataloader position travels in
  `custom_state.json`, and a resumed first epoch skips ahead to that batch.
- **Memory** (`diffsynth/diffusion/runner.py`) — T5 and CLIP stay on the CPU and move to the GPU
  only for a forward, since a 14B DiT plus both encoders exceeds 80GB.
- `grad_norm` logging and clipping.
