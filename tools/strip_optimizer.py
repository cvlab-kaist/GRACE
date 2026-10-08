#!/usr/bin/env python3
"""Shrink a GRACE training checkpoint to the tensors inference actually reads.

A training .ckpt carries the optimizer state and a LoRA copy that generation never
touches. Dropping them is ~4x smaller with bit-identical output.

What is kept, and why — these are not interchangeable:

  ema_state_dict   The weights inference prefers (nested {shadow, shadow_buffers}).
                   kinemadae_video_vae*.py reads this first.
  state_dict       Needed even though EMA wins. With --vae_decoder_checkpoint the loader
                   compares the base's state_dict['gen_model'] against the donor's **bitwise**
                   to find which keys were actually trained. Without it the comparison is
                   vacuous and EVERY donor key gets overlaid - a silently different model.

What is dropped: optimizer_state, lora_state_dict (never read by src/), scaler_state,
sampler_state, and the scalar bookkeeping.

Usage:
    python tools/strip_optimizer.py in.ckpt out.ckpt
"""
import argparse
import os
import sys

import torch

KEEP = ("state_dict", "ema_state_dict")


def nbytes(x):
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(nbytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(nbytes(v) for v in x)
    return 0


def same(a, b):
    """Structural + bitwise equality, so the check cannot pass on a reshaped tensor."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b)
                and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--force", action="store_true", help="overwrite dst if it exists")
    a = ap.parse_args()
    if os.path.realpath(a.src) == os.path.realpath(a.dst):
        sys.exit("[FAIL] src and dst are the same file")
    if os.path.exists(a.dst) and not a.force:
        sys.exit(f"[FAIL] {a.dst} exists (pass --force to overwrite)")

    ckpt = torch.load(a.src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        sys.exit(f"[FAIL] not a checkpoint dict: {type(ckpt).__name__}")
    present = [k for k in KEEP if k in ckpt]
    if "ema_state_dict" not in ckpt and "state_dict" not in ckpt:
        sys.exit("[FAIL] neither ema_state_dict nor state_dict present - nothing to keep")
    if "state_dict" not in ckpt:
        print("[warn] no state_dict: a --vae_decoder_checkpoint overlay against this file "
              "would overlay every donor key. Fine for a base-only run.")

    out = {k: ckpt[k] for k in present}
    torch.save(out, a.dst)

    # Verify by reading back: the kept tensors must be bit-identical.
    back = torch.load(a.dst, map_location="cpu", weights_only=False)
    bad = [k for k in present if not same(ckpt[k], back[k])]
    if bad:
        os.remove(a.dst)
        sys.exit(f"[FAIL] round-trip changed {bad} - output removed")

    si, so = os.path.getsize(a.src), os.path.getsize(a.dst)
    print(f"kept   {', '.join(present)}")
    print(f"dropped {', '.join(k for k in ckpt if k not in present) or '(nothing)'}")
    print(f"{si/1e9:.2f} GB -> {so/1e9:.2f} GB  ({si/so:.1f}x smaller)  verified bit-identical")


if __name__ == "__main__":
    main()
