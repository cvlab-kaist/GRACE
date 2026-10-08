#!/usr/bin/env python3
"""Download the GRACE checkpoints from HuggingFace.

    python tools/download_weights.py              # both tasks
    python tools/download_weights.py --task t2v   # text-to-video only
    python tools/download_weights.py --task i2v   # image-to-video only

Files land in $GRACE_CKPT_DIR (default ./checkpoints) in the layout the launchers
expect, so after this the only thing left to set is where your Wan2.1 base weights are.

This pulls our checkpoints only. The Wan2.1 base models are not redistributed here;
get them from the official release (see README).
"""
import argparse
import os
import sys

REPO = os.environ.get("GRACE_HF_REPO", "chimaharicox/GRACE")
# What each task needs. Keep in sync with scripts/_resolve_paths.sh.
FILES = {
    "t2v": ["t2v/dit.safetensors", "t2v/vae.ckpt", "t2v/decoder.ckpt", "t2v/zmain_stats.json"],
    "i2v": ["i2v/dit.safetensors", "i2v/vae.ckpt", "i2v/decoder.ckpt", "i2v/zmain_stats.json"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("t2v", "i2v", "both"), default="both")
    ap.add_argument("--dest", default=os.environ.get("GRACE_CKPT_DIR", "./checkpoints"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=None, help="pin a commit/tag for reproducibility")
    a = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("[FAIL] pip install huggingface_hub")

    tasks = ["t2v", "i2v"] if a.task == "both" else [a.task]
    want = [f for t in tasks for f in FILES[t]]
    dest = os.path.abspath(a.dest)
    print(f"[grace] {a.repo} -> {dest}")
    print(f"[grace] {len(want)} files for {', '.join(tasks)}. The DiT is tens of GB; this takes a while.")
    for f in want:
        p = hf_hub_download(repo_id=a.repo, filename=f, local_dir=dest, revision=a.revision)
        print(f"  {f:<28} {os.path.getsize(p)/2**30:6.2f} GiB")
    print(f"\n[grace] done. Now:\n  export GRACE_CKPT_DIR={dest}")


if __name__ == "__main__":
    main()
