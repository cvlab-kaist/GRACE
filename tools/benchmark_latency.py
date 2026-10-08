#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure GRACE's generation latency with the released inference code.

The paper's latency table is measured over this window, which is the one the
baselines in the table were measured over too:

    t2v core = first DiT forward of the denoising loop  ->  end of the final VAE decode
    i2v core = start of the first VAE encode            ->  end of the final VAE decode

Checkpoint loading, text encoding and mp4 writing are outside the window for every
model in the table.

The script runs the normal entry point in ../src unchanged and wraps it:
  * setup_davae_dit is wrapped to grab the pipeline object,
  * CUDA events are attached to pipe.vae.encode / pipe.vae.decode and to a forward
    hook on pipe.dit,
  * the first video is a warm-up and is dropped; the median of the rest is reported.

Per video it records core_cuda / core_wall / encode / denoise / decode, the number of
DiT calls and peak memory, and writes them to the output json.

Usage:
    python tools/benchmark_latency.py --task t2v --out result.json -- <args for the entry point>

The defaults of the entry points already match the paper (LoRA folded, DiT resident
during decode, spatially tiled VAE decode).
"""
import argparse, json, os, statistics, sys, time

# The release entry points live next to this file, in ../src.
INF = os.environ.get("GRACE_SRC",
                     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["i2v", "t2v"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    return a, rest


class Timer:
    """영상 1편 단위 상태. CUDA event 쌍 + 호스트 perf_counter."""

    def __init__(self, task, torch):
        self.task, self.torch = task, torch
        self.records = []
        self.reset()

    def reset(self):
        self.enc_start = None; self.enc_end = None; self.enc_calls = 0
        self.dit_first = None; self.dit_last_end = None; self.dit_calls = 0
        self.dec_start = None; self.dec_end = None
        self.wall_start = None
        self.started = False

    def _ev(self):
        e = self.torch.cuda.Event(enable_timing=True); e.record(); return e

    def _begin(self):
        if not self.started:
            self.started = True
            self.torch.cuda.reset_peak_memory_stats()
            self.wall_start = time.perf_counter()

    # --- hooks ---
    def on_encode_start(self):
        self._begin()
        if self.enc_start is None:
            self.enc_start = self._ev()
        self.enc_calls += 1

    def on_encode_end(self):
        self.enc_end = self._ev()

    def on_dit_pre(self):
        self._begin()
        if self.dit_first is None:
            self.dit_first = self._ev()
        self.dit_calls += 1

    def on_dit_post(self):
        self.dit_last_end = self._ev()

    def on_decode_start(self):
        self.dec_start = self._ev()

    def on_decode_end(self):
        self.dec_end = self._ev()
        self.torch.cuda.synchronize()
        wall = time.perf_counter() - self.wall_start
        core_start = self.enc_start if self.task == "i2v" else self.dit_first
        if core_start is None:   # i2v 인데 encode 가 안 불린 경우 등 — DiT 기준으로 폴백
            core_start = self.dit_first
        ms = lambda a, b: (a.elapsed_time(b) / 1e3) if (a is not None and b is not None) else None
        rec = dict(
            core_cuda_s=ms(core_start, self.dec_end),
            core_wall_s=wall,
            encode_s=ms(self.enc_start, self.enc_end), encode_calls=self.enc_calls,
            denoise_s=ms(self.dit_first, self.dit_last_end), dit_calls=self.dit_calls,
            decode_s=ms(self.dec_start, self.dec_end),
            gap_s=None,
            max_allocated_gib=self.torch.cuda.max_memory_allocated() / 2**30,
            max_reserved_gib=self.torch.cuda.max_memory_reserved() / 2**30,
        )
        parts = [x for x in (rec["encode_s"] if self.task == "i2v" else None, rec["denoise_s"], rec["decode_s"]) if x]
        rec["gap_s"] = rec["core_cuda_s"] - sum(parts)   # 경계 안인데 stage 밖인 시간(offload 왕복 등)
        self.records.append(rec)
        n = len(self.records)
        print(f"[e2e] #{n} core_cuda {rec['core_cuda_s']:.2f}s wall {rec['core_wall_s']:.2f}s | "
              f"enc {rec['encode_s'] or 0:.2f} (x{rec['encode_calls']}) denoise {rec['denoise_s']:.2f} "
              f"(DiT fwd x{rec['dit_calls']}) dec {rec['decode_s']:.2f} gap {rec['gap_s']:.2f} | "
              f"peak {rec['max_allocated_gib']:.1f} GiB", flush=True)
        self.reset()


def merge_lora(dit):
    from peft.tuners.lora import LoraLayer
    merged = 0
    for m in dit.modules():
        if isinstance(m, LoraLayer):
            m.merge(); merged += 1
    swapped = 0
    for parent in dit.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoraLayer):
                setattr(parent, name, child.base_layer); swapped += 1
    left = sum(1 for m in dit.modules() if isinstance(m, LoraLayer))
    assert left == 0, f"LoRA 래퍼 {left}개 잔존"
    n = sum(p.numel() for p in dit.parameters())
    print(f"[e2e] LoRA merge: fold {merged}, 래퍼 제거 {swapped}, params {n/1e9:.2f}B", flush=True)


def main():
    a, rest = parse()
    sys.path.insert(0, INF)
    os.chdir(INF)
    import torch
    import inference_i2v_geoprior_crossattn as MI
    M = MI if a.task == "i2v" else __import__("inference_t2v_geoprior")
    T = Timer(a.task, torch)
    merge = os.environ.get("GRACE_MERGE_LORA", "1") == "1"
    offload = os.environ.get("DIT_OFFLOAD_DECODE", "1") != "0"
    print(f"[e2e] task={a.task} merge_lora={merge} dit_offload_decode={offload} "
          f"gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    _orig_setup = MI.setup_davae_dit

    def _setup(pipe, *args, **kw):
        r = _orig_setup(pipe, *args, **kw)
        if merge:
            merge_lora(pipe.dit)
        # --- hooks ---
        pipe.dit.register_forward_pre_hook(lambda m, i: T.on_dit_pre())
        pipe.dit.register_forward_hook(lambda m, i, o: T.on_dit_post())
        _enc, _dec = pipe.vae.encode, pipe.vae.decode

        def enc(*x, **k):
            T.on_encode_start(); out = _enc(*x, **k); T.on_encode_end(); return out

        def dec(*x, **k):
            T.on_decode_start(); out = _dec(*x, **k); T.on_decode_end(); return out
        pipe.vae.encode, pipe.vae.decode = enc, dec
        print("[e2e] hooks installed (dit fwd / vae.encode / vae.decode)", flush=True)
        return r
    MI.setup_davae_dit = _setup
    M.setup_davae_dit = _setup      # t2v 는 from-import 로 이름을 복사해 갖고 있다

    sys.argv = [M.__file__] + rest
    t0 = time.time()
    M.main()
    total = time.time() - t0

    recs = T.records
    timed = recs[a.warmup:]
    med = lambda k: statistics.median([r[k] for r in timed if r[k] is not None]) if timed else None
    summary = dict(
        task=a.task, n_videos=len(recs), warmup=a.warmup, n_timed=len(timed),
        merge_lora=merge, dit_offload_decode=offload,
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
        boundary=("첫 VAE encode 시작 → 최종 VAE decode 끝" if a.task == "i2v"
                  else "첫 DiT forward → 최종 VAE decode 끝"),
        median=dict(core_cuda_s=med("core_cuda_s"), core_wall_s=med("core_wall_s"),
                    encode_s=med("encode_s"), denoise_s=med("denoise_s"), decode_s=med("decode_s"),
                    gap_s=med("gap_s"), max_allocated_gib=med("max_allocated_gib")),
        per_video=recs, release_args=rest, total_wall_incl_load_s=total,
        env={k: os.environ.get(k) for k in ("GRACE_MERGE_LORA", "DIT_OFFLOAD_DECODE", "CUDA_VISIBLE_DEVICES")},
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1, ensure_ascii=False)
    m = summary["median"]
    print(f"\n[e2e] {a.task} median over {len(timed)} timed: core_cuda {m['core_cuda_s']:.2f}s "
          f"wall {m['core_wall_s']:.2f}s (enc {m['encode_s'] or 0:.2f} denoise {m['denoise_s']:.2f} "
          f"dec {m['decode_s']:.2f} gap {m['gap_s']:.2f}) peak {m['max_allocated_gib']:.1f} GiB → {a.out}", flush=True)


if __name__ == "__main__":
    main()
