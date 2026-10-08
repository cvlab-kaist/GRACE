#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GRACE text-to-video inference.

Differences from the image-to-video entry point (inference_i2v_geoprior_crossattn.py), all verified
against the training runs:

     #  item                    i2v                                t2v (this file)
     1  dual timestep           --async_delta 0.15 --shift_main 3   not passed
     2  patchify input channels 72                                  32
     3  channel layout          (main16+prior16)x2 + mask8          main16+prior16 (noisy only)
     4  LoRA targets            q,k,v,o,k_img,v_img,ffn.0,ffn.2     q,k,v,o,ffn.0,ffn.2
     5  DiT                     Wan2.1-I2V-14B-480P (7 shards)      Wan2.1-T2V-14B (6 shards)
     6  CLIP                    required                            not used
     7  input image             --image_dir / ff_inject / ff_window none
     8  VAE loader              crossattn (first_frame_inject)      plain load_kinemadae_geoprior_vae

Item 4 is the dangerous one. Leaving k_img and v_img in place targets modules the t2v DiT does not
have, PEFT injection goes wrong and strict=False swallows it silently - once seen as 160 dropped keys
that only showed up as swirl artifacts. The dropped keys are printed after loading for that reason.

Item 1: with setup_davae_dit(use_async=False) the t2_projection module is never built and the two
checkpoint keys are dropped on purpose. In a pure t2v checkpoint those weights are |w|=0 (never
trained), so nothing is lost. wan_video.py gates on `timestep2 is not None and
hasattr(dit, 't2_projection')`, so not passing timestep2 keeps the single-timestep path.

Usage
  python inference_t2v_geoprior.py --dit_checkpoint <step-N.safetensors> --out_dir <dir> --dry_run
  python inference_t2v_geoprior.py --dit_checkpoint <...> --out_dir <dir> --dims subject_consistency --per_dim 1
"""
import os, sys, json, re, random, argparse, time
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

# [release] The R2n autoencoder definition is bundled at src/train_crossattn/ - the exact copy the
#   released checkpoints were trained with. An older revision of the same file loads far enough to
#   look fine and then fails in stages_norm_before_head, so the path is pinned rather than searched.
#   It must be set before kinemadae_video_vae_crossattn is imported; that module reads it at import.
os.environ.setdefault(
    "KINEMADAE_CROSSATTN_REPO",
    os.path.join(_THIS, "train_crossattn"),
)
# ** DIFFSYNTH_ROOT - if this is not pinned here you silently get the wrong videos (it happened once).
#   When the path does not exist the sys.path insert is a no-op and Python picks up whichever
#   other DiffSynth copy it finds first. The upstream copy has no rope_pos_scale support, so the
#   assignment to pipe.dit.rope_pos_scale is quietly ignored and sampling runs with plain rope -
#   latents that do not match training (2,2,2), with no error. Hence the bundled copy is pinned.
#   Setting it only in the launcher brings the bug back when python is called directly, so it lives here.
os.environ.setdefault(
    "DIFFSYNTH_ROOT",
    os.path.join(os.path.dirname(_THIS), "third_party", "DiffSynth-Studio"),
)

# Reused by the i2v module - its only import-time side effect is the sys.path insert, so this is safe.
#   setup_davae_dit already has the `_has_image=False` branch (t2v, 32-channel patchify).
from inference_i2v_geoprior_crossattn import (          # noqa: E402
    setup_davae_dit, KinemaDAENoiseInitializer, _ROOT, fold_lora, _save,
)
# [release] The only mp4 writer is _save() in the i2v module. The quality switches are globals there,
#   so the module object is held here and assigned in main() rather than duplicating the writer.
import inference_i2v_geoprior_crossattn as _i2v_mod      # noqa: E402
import diffsynth.pipelines.wan_video as _wv               # noqa: E402
# * Pin which copy was actually loaded, and check that it **consumes** rope_pos_scale.
#   Assigning pipe.dit.rope_pos_scale succeeds on any copy, so a 'set it' log proves nothing.
#   Without the consumer (:1463 `getattr(dit, "rope_pos_scale", None)`) the value is silently ignored.
_WV_SRC = os.path.abspath(_wv.__file__)
with open(_WV_SRC, encoding="utf-8") as _f:
    _WV_HAS_ROPE = "rope_pos_scale" in _f.read()
print(f"[diffsynth] {_WV_SRC}\n            rope_pos_scale applied={_WV_HAS_ROPE}")
if not _WV_HAS_ROPE:
    raise SystemExit(
        "[FAIL] The DiffSynth copy that was loaded has no rope_pos_scale support, so\n"
        "       --rope_pos_scale is silently ignored and the latents will not match\n"
        "       training. Point DIFFSYNTH_ROOT at third_party/DiffSynth-Studio in this repo.\n"
        f"       Loaded: {_WV_SRC}")
from diffsynth.pipelines.wan_video import (              # noqa: E402
    WanVideoPipeline, ModelConfig,
    WanVideoUnit_NoiseInitializer, WanVideoUnit_ImageEmbedderVAE,
)
# * The plain `kinemadae_video_vae` in this repo does **not** support R2n (no stages_after_norm argument).
#   Only the crossattn variant takes the R2n arguments. With first_frame_inject=False the ff path is off,
#   which gives the same configuration as plain geoprior. (t2v has no first frame, so ff must stay off.)
from kinemadae_video_vae_crossattn import (                  # noqa: E402
    load_kinemadae_geoprior_vae_crossattn as load_geoprior_vae,
)


# ---------------------------------------------------------------------------
# T2V dual-timestep sampling - only for checkpoints trained with the asynchronous delta band.
#
# Why it is needed
#   The old path sets use_async=False, never builds t2_projection and drops the two checkpoint keys.
#   That is only correct for a **pure** run (t2_projection |w|=0, never trained).
#   An async run is trained to |w|=4.2e-02, so dropping those keys throws away the whole
#   base-leading behaviour and samples as if it were pure - without any error.
#
# What is reused (nothing is reimplemented here)
#   async_ladders / dual_step  : the shared implementation in dual_sched_core.py. The i2v path calls
#                                the **same functions**, so ladder geometry and Euler steps are bit identical.
#   pipe.unit_runner / model_fn: the standard pipeline path.
#
# How this differs from the i2v version (inference_i2v_geoprior_crossattn.py:367)
#   - no input_image / y / CLIP / first_frame_inject (t2v has no conditioning path)
#   - no FrameInit, prior_cfg, restart or inject_prior_latents (i2v-only experiments)
#   => what is left is two ladders, passing timestep2, and a Euler step per branch.
# ---------------------------------------------------------------------------
def dual_schedule_generate_t2v(pipe, prompt, negative_prompt,
                               height, width, num_frames, num_inference_steps, seed,
                               shift_main, z_dim, async_delta,
                               cfg_scale=5.0, tiled=True,
                               tile_size=(30, 52), tile_stride=(15, 26),
                               async_shift_prior=None, prior_sigma_floor=0.02):
    import torch as _t
    from dual_sched_core import async_ladders, dual_step

    dev, dt = pipe.device, pipe.torch_dtype
    dit = pipe.dit
    assert hasattr(dit, "t2_projection"), (
        "[FAIL] no t2_projection - build the DiT with setup_davae_dit(use_async=True)")

    # The trunk clock is t_main. Pipeline units read pipe.scheduler, so it is set first.
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=shift_main)

    inputs_posi = {"prompt": prompt, "vap_prompt": None,
                   "tea_cache_l1_thresh": None, "tea_cache_model_id": "",
                   "num_inference_steps": num_inference_steps}
    inputs_nega = {"negative_prompt": negative_prompt, "negative_vap_prompt": None,
                   "tea_cache_l1_thresh": None, "tea_cache_model_id": "",
                   "num_inference_steps": num_inference_steps}
    inputs_shared = {
        "input_image": None, "end_image": None,
        "input_video": None, "denoising_strength": 1.0,
        "control_video": None, "reference_image": None,
        "camera_control_direction": None, "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0,
                                  0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None, "vace_video_mask": None, "vace_reference_image": None, "vace_scale": 1.0,
        "seed": seed, "rand_device": "cpu",
        "height": height, "width": width, "num_frames": num_frames,
        "cfg_scale": cfg_scale, "cfg_merge": False,
        "sigma_shift": shift_main,
        "motion_bucket_id": None, "longcat_video": None,
        "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        "sliding_window_size": None, "sliding_window_stride": None,
        "input_audio": None, "audio_sample_rate": 16000, "s2v_pose_video": None,
        "audio_embeds": None, "s2v_pose_latents": None, "motion_video": None,
        "animate_pose_video": None, "animate_face_video": None,
        "animate_inpaint_video": None, "animate_mask_video": None,
        "vap_video": None,
        "wantodance_music_path": None, "wantodance_reference_image": None, "wantodance_fps": 30,
        "wantodance_keyframes": None, "wantodance_keyframes_mask": None,
        "framewise_decoding": False,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    # Two ladders - same geometry as training (_async_flow_match_loss)
    sig_main, sig_prior = async_ladders(num_inference_steps, shift_main, async_delta,
                                        shift_prior=async_shift_prior,
                                        sigma_min=prior_sigma_floor)
    if prior_sigma_floor > 0.0:
        sig_prior = sig_prior.clamp(min=prior_sigma_floor)
    sig_main, sig_prior = sig_main.to(dev), sig_prior.to(dev)
    ts_main, ts_prior = sig_main * 1000.0, sig_prior * 1000.0
    print(f"[dual-t2v] Δ={async_delta} shift_main={shift_main} "
          f"σ_main {float(sig_main[0]):.3f}→{float(sig_main[-1]):.3f} / "
          f"σ_prior {float(sig_prior[0]):.3f}→{float(sig_prior[-1]):.3f}", flush=True)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for i in range(num_inference_steps):
        t_main = ts_main[i:i + 1].to(dtype=dt, device=dev)
        t_prior = ts_prior[i:i + 1].to(dtype=dt, device=dev)
        # model_fn handles both the t2_projection injection and the head relay for timestep2
        # (wan_video.py:1394-1399). Touching dit.head.t_prior_emb here would set it twice.
        v_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi,
                               timestep=t_main, timestep2=t_prior)
        if cfg_scale != 1.0:
            v_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega,
                                   timestep=t_main, timestep2=t_prior)
            v = v_nega + cfg_scale * (v_posi - v_nega)
        else:
            v = v_posi
        inputs_shared["latents"] = dual_step(inputs_shared["latents"], v, i,
                                             sig_main, sig_prior, z_dim)
    dit.head.t_prior_emb = None   # reset state so it does not leak into the next prompt

    video = pipe.vae.decode(inputs_shared["latents"], device=dev, tiled=False)
    return pipe.vae_output_to_video(video)


NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# [release] The VBench prompt set the paper reports ships with this repo under assets/vbench,
#   so a benchmark run needs no setup. GRACE_VBENCH_JSON / GRACE_VBENCH_AUG_DIR /
#   GRACE_VBENCH_ORIG_DIR still override, e.g. to point at your own VBench checkout.
_ASSETS = os.path.join(os.path.dirname(_THIS), "assets", "vbench")
VBENCH_JSON = os.environ.get("GRACE_VBENCH_JSON", os.path.join(_ASSETS, "VBench_full_info.json"))


# ---------------------------------------------------------------------------
def build_model_paths_t2v(dit_dir, shared_dir):
    """T2V: 6 DiT shards + T5 + the Wan VAE. No CLIP (has_image_input=False).

    T5, the VAE and the tokenizer are shared from the I2V-480P folder, the same as in training
    (the t2v launcher sets CKPT_DIR=Wan2.1-T2V-14B, SHARED_DIR=Wan2.1-I2V-14B-480P).
    Wan2.1-T2V-14B has no per-resolution variant - one model covers 480p and 720p.
    """
    shards = [os.path.join(dit_dir, f"diffusion_pytorch_model-0000{i}-of-00006.safetensors")
              for i in range(1, 7)]
    for p in shards:
        assert os.path.exists(p), f"[FAIL] missing T2V shard: {p}"
    return [
        shards,
        os.path.join(shared_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(shared_dir, "Wan2.1_VAE.pth"),
    ]


def load_vbench_by_dim(path):
    """VBench_full_info.json -> {dimension: [prompt_en, ...]}.

    946 entries, 944 unique prompts. A prompt can belong to several dimensions, so the per-dimension
    lists overlap (subject_consistency, dynamic_degree and motion_smoothness share the same 72).
    """
    with open(path) as f:
        data = json.load(f)
    by = {}
    for x in data:
        for d in x["dimension"]:
            by.setdefault(d, []).append(x["prompt_en"])
    # de-duplicate within a dimension, keeping the original order
    return {k: list(dict.fromkeys(v)) for k, v in sorted(by.items())}


VBENCH_AUG_DIR = os.environ.get(
    "GRACE_VBENCH_AUG_DIR", os.path.join(_ASSETS, "prompts", "prompts_per_dimension_FINAL736"))
VBENCH_ORIG_DIR = os.environ.get(
    "GRACE_VBENCH_ORIG_DIR", os.path.join(_ASSETS, "prompts", "prompts_per_dimension"))


def apply_augmented_prompts(by_dim, aug_dir=None, orig_dir=None, verbose=True):
    """Replace the short VBench prompts with their gpt_enhanced expanded versions.

    Why match by index: the expanded text does not contain the original verbatim, it dissolves it.
      original  "a person swimming in ocean"
      expanded  "A lone swimmer, clad in a sleek black wetsuit, glides effortlessly ..."
    So string matching cannot find it. The per-dimension order in VBench_full_info.json and the line
    order in prompts_per_dimension/<dim>.txt were verified to match exactly (subject_consistency
    72/72, human_action 100/100, object_class 79/79), so the same line number is the same prompt.

    Safety: before substituting, the original txt and the json order are checked every time. If they
    disagree the dimension is left **unsubstituted** - generating from the wrong prompt without
    noticing is the worst outcome.
    Dimensions with no expanded file (dynamic_degree and others) keep the originals.
    """
    import os as _os
    aug_dir = aug_dir or VBENCH_AUG_DIR
    orig_dir = orig_dir or VBENCH_ORIG_DIR
    out, stat = {}, []
    for dim, prompts in by_dim.items():
        fa = _os.path.join(aug_dir, f"{dim}_longer.txt")
        fo = _os.path.join(orig_dir, f"{dim}.txt")
        if not (_os.path.exists(fa) and _os.path.exists(fo)):
            out[dim] = prompts; stat.append((dim, "no augmented file -> original", len(prompts))); continue
        aug = [l.strip() for l in open(fa, encoding="utf-8") if l.strip()]
        org = [l.strip() for l in open(fo, encoding="utf-8") if l.strip()]
        n = min(len(prompts), len(org), len(aug))
        mism = sum(1 for i in range(n) if prompts[i].strip() != org[i])
        if mism:
            out[dim] = prompts
            stat.append((dim, f"order mismatch ({mism}) -> keeping original", len(prompts))); continue
        out[dim] = [aug[i] if i < len(aug) else prompts[i] for i in range(len(prompts))]
        stat.append((dim, "replaced", n))
    if verbose:
        print("  [prompt] applying gpt_enhanced augmented prompts")
        for d, how, n in stat:
            print(f"    {d:<24} {how} ({n})")
    # [release] The default is to keep the original when substitution fails - safer than generating
    #   from the wrong sentence. But a run that measures the paper's numbers must not fall back
    #   silently, or the result is not comparable.
    if os.environ.get("GRACE_REQUIRE_AUGMENTED") == "1":
        bad = [(d, how) for d, how, _ in stat if how != "replaced"]
        assert not bad, (
            "[FAIL] GRACE_REQUIRE_AUGMENTED=1 but these dimensions were not augmented: "
            + ", ".join(f"{d}({how})" for d, how in bad))
    return out


def safe_name(prompt, maxlen=90):
    s = re.sub(r"[^A-Za-z0-9가-힣 _-]", "", prompt).strip().replace(" ", "_")
    return (s[:maxlen] or "prompt")


def sample_plan(by_dim, dims, per_dim, seed):
    """Sample each dimension independently. Dimensions share prompts, so the same sentence can
    appear in two folders; each folder then gets its own generation, which keeps every folder
    self-contained and matches the VBench evaluation convention."""
    plan = {}
    for d in dims:
        pool = by_dim[d]
        n = min(per_dim, len(pool))
        plan[d] = random.Random(f"{seed}:{d}").sample(pool, n)
    return plan


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit_checkpoint", required=True,
                    help="trained GRACE checkpoint for the DiT")
    ap.add_argument("--out_dir", required=True)
    # model assets (same as the t2v training launcher)
    ap.add_argument("--dit_dir",    default=os.environ.get("GRACE_WAN_T2V_DIR", ""))
    ap.add_argument("--shared_dir", default=os.environ.get("GRACE_WAN_SHARED", os.environ.get("GRACE_WAN_I2V_DIR", "")))
    # * Must be the same file as the stage-1 VAE checkpoint used in training (STAGE1_RES in the training sbatch).
    ap.add_argument("--vae_checkpoint", default=os.environ.get("GRACE_VAE_CKPT", ""))
    ap.add_argument("--vae_pretrained", default=None, help="default: <shared_dir>/Wan2.1_VAE.pth")
    ap.add_argument("--zmain_stats_path", default=os.environ.get("GRACE_ZMAIN_STATS", ""))
    ap.add_argument("--vae_z_dim", type=int, default=16)
    ap.add_argument("--vae_prior_z_dim", type=int, default=16)
    ap.add_argument("--vae_prior_subsample_mode", default="bilinear",
                    help="must match the value the VAE was trained with")
    # [decoder-swap] base (encoder, prior, EMA) comes from --vae_checkpoint; only the **trained decoder** from this one.
    #   The loader compares donor and base key by key and loads only the keys that differ - so the donor must be
    #   a decoder-only run whose encoder stayed frozen, otherwise every key differs and this becomes a full VAE
    #   swap. Before using it, check that the donor's --init_vae_from matches --vae_checkpoint.
    #   (Even in decoder-only runs the EMA encoder drifts toward raw, so never load the whole VAE.)
    ap.add_argument("--vae_decoder_checkpoint", default=None,
                    help="decoder-only checkpoint to overlay on top of the VAE")
    ap.add_argument("--save_latents", action="store_true",
                    help="also write the final latent next to the mp4 as _latents.pt")
    # LoRA - note there is no k_img/v_img for t2v
    ap.add_argument("--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")
    # Only for checkpoints trained with the asynchronous delta band. Given, sampling takes the dual-timestep path.
    #   Left unset (None) it is the old pure path - no change in behaviour.
    ap.add_argument("--async_delta", type=float, default=None,
                    help="asymmetric denoising offset delta, in u space. "
                         "0 denoises the base and residual latents together")
    ap.add_argument("--async_shift_prior", type=float, default=None,
                    help="warp only the base ladder with a different shift; normally left unset")
    # [default 3.0] The i2v argparse default says 5.0, but that is only the 'baseline' label;
    #   the actual i2v runs use SHIFT_MAIN=3 (measured across the session scripts: 3 in 63 runs, 2 in 7).
    #   The comparison table at the top of this file also lists i2v as '--async_delta 0.15 --shift_main 3'.
    #   Note: runs trained with shift sampled from U(2,5) make a 2-5 sweep meaningful.
    ap.add_argument("--shift_main", type=float, default=3.0,
                    help="beta_main, the flow-matching shift of the main ladder")
    ap.add_argument("--lora_rank", type=int, default=512)
    ap.add_argument("--rope_pos_scale", default="2,2,2")
    # sampling
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--spatial_tile", action="store_true", default=True,
                    help="tile the VAE decode spatially; on by default because a single pass at 480x832x81 peaks near 118 GB")
    ap.add_argument("--no_spatial_tile", dest="spatial_tile", action="store_false")
    ap.add_argument("--spatial_tile_size",   default="20,28")
    # Same as the i2v default (:752) - a 20x28 tile with 10x14 stride is 50% overlap. The real i2v runs used
    # these values, so seamlessness is already verified. Less overlap is faster but leaves the verified setting.
    ap.add_argument("--spatial_tile_stride", default="10,14")
    # VBench sampling
    # [release] Single-prompt mode - one video, no VBench json. It goes through the same pipeline and the
    #   same writer as the benchmark path, so the result equals one benchmark cell.
    ap.add_argument("--prompt", default="",
                    help="a prompt, or a file with one prompt per line (blank lines and lines "
                         "starting with # are skipped). A file generates every prompt in a single "
                         "process, so the model is loaded once. No VBench json needed.")
    ap.add_argument("--no_merge_lora", action="store_true",
                    help="keep the LoRA wrappers instead of folding them into the base weights. "
                         "Folding is the default: it is how the latency in the paper was measured "
                         "and what you want for normal use. Folding is NOT bit-exact in bf16, "
                         "though (measured 42.3 dB on one 480x832x81 sample), so pass this flag to "
                         "reproduce the exact videos the paper's VBench scores came from.")
    ap.add_argument("--maxq", action="store_true", default=True,
                    help=argparse.SUPPRESS)        # default; kept so old command lines still work
    ap.add_argument("--small", dest="maxq", action="store_false",
                    help="write smaller, lossy mp4s instead of the default visually lossless ones")
    ap.add_argument("--lossless", action="store_true",
                    help="bit-exact RGB (libx264rgb crf 0, rgb24). FOR METRICS ONLY - gbrp streams "
                         "render with a green cast in browsers and most players. Use --maxq to watch.")
    ap.add_argument("--vbench_json", default=VBENCH_JSON)
    ap.add_argument("--prompt_map", default="",
                    help="JSON {dimension: {idx: prompt}} overriding planned prompts at those "
                         "indices. The plan (idx -> caption slot) is untouched, so scoring stays "
                         "aligned; only the text sent to the model changes. Used to reproduce the "
                         "paper's finalized prompt set.")
    ap.add_argument("--per_dim", type=int, default=5)
    ap.add_argument("--only_idx", default="",
                    help="generate only these plan indices (e.g. 26,42,55). The plan itself is "
                         "unchanged, so idx-to-prompt stays the same; changing --per_dim or "
                         "--seed would shift it")
    ap.add_argument("--dims", default="", help="comma separated; all 16 dimensions if unset")
    ap.add_argument("--augmented_prompts", action="store_true",
                    help="use the gpt_enhanced expanded VBench prompts instead of the short "
                         "originals")
    ap.add_argument("--aug_prompt_dir", default=VBENCH_AUG_DIR)
    ap.add_argument("--orig_prompt_dir", default=VBENCH_ORIG_DIR)
    ap.add_argument("--dry_run", action="store_true", help="build the plan, folders and manifest without loading the model")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.vae_pretrained is None:
        args.vae_pretrained = os.path.join(args.shared_dir, "Wan2.1_VAE.pth")
    _i2v_mod._LOSSLESS = bool(args.lossless)   # [release] hand the save-quality switches to _save()
    _i2v_mod._MAXQ = bool(args.maxq)
    t_start = time.time()

    # -- 1) prompt plan ---------------------------------------------------
    if args.prompt:
        # [release] --prompt takes one sentence, or a file with one prompt per line.
        #   With a file the model is loaded **once** and everything is generated (loading costs more than sampling).
        #   It is turned into a one-dimension plan ('sample') so the loop below is reused unchanged.
        if os.path.isfile(args.prompt):
            with open(args.prompt, encoding="utf-8") as f:
                prompts = [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]
            assert prompts, f"[FAIL] prompt file is empty: {args.prompt}"
            print(f"[plan] prompt file {args.prompt} -> {len(prompts)} videos  (seed={args.seed})")
        else:
            prompts = [args.prompt]
            print(f"[plan] single prompt, 1 video  (seed={args.seed})\n   {args.prompt[:120]}")
        dims = ["sample"]
        plan = {"sample": prompts}
        by_dim = dict(plan)
    else:
        assert args.vbench_json, (
            "[FAIL] neither --prompt nor --vbench_json was given. Pass --prompt \"...\" for a "
            "single video, or --vbench_json (or GRACE_VBENCH_JSON) to run the whole benchmark")
        by_dim = load_vbench_by_dim(args.vbench_json)
        if args.augmented_prompts:
            by_dim = apply_augmented_prompts(by_dim, args.aug_prompt_dir, args.orig_prompt_dir)
        dims = [d.strip() for d in args.dims.split(",") if d.strip()] or list(by_dim)
        unknown = [d for d in dims if d not in by_dim]
        assert not unknown, f"[FAIL] unknown dimension: {unknown}\n  available: {list(by_dim)}"
        plan = sample_plan(by_dim, dims, args.per_dim, args.seed)
        if args.prompt_map:
            # The idx-to-slot mapping is left alone and only the sentence in that slot is swapped. The filename is
            # safe_name(new sentence), so a changed entry lands in a new file (identical first 90 characters give the
            # same name, so delete the old file before regenerating or it will be skipped - this happened once).
            with open(args.prompt_map) as f:
                pm = json.load(f)
            nsw = 0
            for d, m in pm.items():
                if d not in plan:
                    continue
                for k, newp in m.items():
                    i = int(k)
                    if 0 <= i < len(plan[d]) and plan[d][i] != newp:
                        plan[d][i] = newp; nsw += 1
            print(f"[prompt_map] {args.prompt_map} -> {nsw} entries replaced")
        total = sum(len(v) for v in plan.values())
        print(f"[plan] {len(dims)} dimensions x up to {args.per_dim} = {total} videos  (seed={args.seed})")
        for d in dims:
            print(f"   {d:24s} pool {len(by_dim[d]):3d} -> {len(plan[d])}")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {}
    for d in dims:
        _sub = "" if args.prompt else d        # with a single prompt, write straight into out_dir
        os.makedirs(os.path.join(args.out_dir, _sub), exist_ok=True)
        manifest[d] = [{"idx": i, "prompt": p, "seed": args.seed,
                        "path": os.path.join(_sub, f"{i:02d}_{safe_name(p)}.mp4")}
                       for i, p in enumerate(plan[d])]
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"dit_checkpoint": args.dit_checkpoint, "seed": args.seed,
                   "height": args.height, "width": args.width, "num_frames": args.num_frames,
                   "num_inference_steps": args.num_inference_steps, "cfg_scale": args.cfg_scale,
                   "dims": manifest}, f, indent=2, ensure_ascii=False)
    print(f"[plan] manifest → {os.path.join(args.out_dir,'manifest.json')}")
    if args.dry_run:
        print("[dry_run] exiting without loading the model"); return

    # -- 2) pipeline ------------------------------------------------------
    print("\n[load] T2V pipeline (no CLIP)")
    model_paths = build_model_paths_t2v(args.dit_dir, args.shared_dir)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=args.device,
        model_configs=[ModelConfig(p) for p in model_paths],
        tokenizer_config=ModelConfig(os.path.join(args.shared_dir, "google", "umt5-xxl")),
    )
    _has_img = getattr(pipe.dit, "has_image_input", False)
    assert not _has_img, ("[FAIL] has_image_input=True - an I2V DiT was loaded. Pass the T2V "
                          "shards (Wan2.1-T2V-14B) so patchify is built with 32 input channels")
    print(f"  has_image_input={_has_img} (T2V confirmed)")

    # -- 3) geoprior VAE - same plain loader and arguments as training (train_dit.py:1597) --
    print("\n[load] Geoprior VAE (plain, not crossattn/ff)")
    pipe.vae = load_geoprior_vae(
        checkpoint_path=args.vae_checkpoint,
        pretrained_path=args.vae_pretrained,
        z_dim=args.vae_z_dim, prior_z_dim=args.vae_prior_z_dim,
        add_encoder_stages=[{"mode": "downsample3d", "num_res_blocks": 2, "init": "zero"}],
        add_decoder_before_head_stages=[{"mode": "upsample3d", "num_res_blocks": 2}],
        no_expand_conv2=True,            # training --vae_no_expand_conv2
        stages_after_norm=True,          # training --vae_stages_after_norm
        stages_norm_before_head=True,    # training --vae_stages_norm_before_head
        expand_encoder_head=False,       # training did not pass --vae_expand_encoder_head
        decoder_mirror=False,            # training --vae_no_decoder_mirror
        # * Training used --vae_prior_subsample_mode bilinear. This loader defaults to avg_pool, so it must be explicit.
        #   (t2v sampling never runs encode, but loading a mismatched configuration invites confusion later.)
        subsample_mode=args.vae_prior_subsample_mode,
        first_frame_inject=False,        # * t2v - there is no first frame
        decoder_checkpoint_path=args.vae_decoder_checkpoint,   # None means no swap (the old behaviour)
        device=args.device, dtype=torch.bfloat16,
        zmain_stats_path=args.zmain_stats_path, zprior_norm=True,
    )
    pipe.vae.requires_grad_(False); pipe.vae.eval()
    pipe.vae._use_spatial_tile = args.spatial_tile
    pipe.vae._spatial_tile_size   = tuple(int(x) for x in args.spatial_tile_size.split(","))
    pipe.vae._spatial_tile_stride = tuple(int(x) for x in args.spatial_tile_stride.split(","))
    _lat_ch = getattr(pipe.vae, "latent_channels", None)
    print(f"  latent_channels={_lat_ch}  temporal_factor={pipe.vae.temporal_factor}"
          f"  upsampling_factor={pipe.vae.upsampling_factor}  spatial_tile={args.spatial_tile}")
    assert _lat_ch == args.vae_z_dim + args.vae_prior_z_dim, \
        f"[FAIL] latent_channels={_lat_ch} != {args.vae_z_dim}+{args.vae_prior_z_dim}"

    # -- 3b) replace the decode path - the decode inside pipe() is single-pass and runs out of memory --
    #   pipe.decode_video follows the Wan convention and calls vae.decode(..., tiled=True, tile_size=(30,52)).
    #   But at 480x832 the latent grid is exactly 30x52, so 'one tile' is the whole frame - effectively a single
    #   pass, and one decoder upsample was measured asking for 23.1 GiB and running out of memory (80 GB card,
    #   DiT resident). This swaps in the spatial_tiled path the i2v version uses (:684-690). Its argument names
    #   differ from tiled/tile_size, so pipe cannot pass them through and the bound method is wrapped.
    #   The DiT (~28 GB) is also moved to the CPU during decode, for the same reason as the i2v version (:676-680).
    #   * Unlike the i2v version, it is moved back **explicitly** in a finally. pipe.load_models_to_device is
    #     a no-op without vram_management, so there is no basis for expecting an automatic return (measured).
    #   * This wrapper is also where --save_latents lives. pipe() returns finished frames and no latent, but the
    #     hidden_states entering decode are exactly that final latent. The same thing is obtained without
    #     duplicating the denoise loop the way the i2v version does.
    _vae_decode_orig = pipe.vae.decode
    _dev_for_dit = args.device
    _cap = {"latents": None}          # holds the final latent that arrived at decode

    def _decode_hook(hidden_states, device, **kw):
        kw.pop("tiled", None); kw.pop("tile_size", None); kw.pop("tile_stride", None)
        if args.save_latents:
            #  (B,32,T,h,w) - the normalized model space as is. decode_prior_only.py takes [:,16:] and feeds it to the
            #  original Wan decoder; zprior_norm was normalized with Wan's own latent statistics, so the Wan decoder
            #  denormalizes it internally - no extra conversion needed.
            _cap["latents"] = hidden_states.detach().to("cpu", torch.float32).clone()
        # [release] By default the DiT stays on the GPU during decode - the state the paper's latency was measured in.
        #   Set DIT_OFFLOAD_DECODE=1 if VRAM is tight; the output is the same either way (only weight placement
        #   changes). Turning it on adds about 5 s per video for the 14B round trip.
        _off = os.environ.get("DIT_OFFLOAD_DECODE", "0") != "0"
        if _off:
            pipe.dit.to("cpu"); torch.cuda.empty_cache()
        try:
            if args.spatial_tile:
                return _vae_decode_orig(
                    hidden_states, device,
                    spatial_tiled=True,
                    spatial_tile_size=pipe.vae._spatial_tile_size,
                    spatial_tile_stride=pipe.vae._spatial_tile_stride,
                    **kw)
            return _vae_decode_orig(hidden_states, device, tiled=False, **kw)
        finally:
            if _off:
                pipe.dit.to(_dev_for_dit)

    pipe.vae.decode = _decode_hook
    print(f"  [decode] spatial_tile={args.spatial_tile} "
          f"tile={pipe.vae._spatial_tile_size} stride={pipe.vae._spatial_tile_stride}, "
          f"DiT offload={os.environ.get('DIT_OFFLOAD_DECODE', '0') != '0'}, "
          f"save_latents={args.save_latents}")

    # -- 4) pipeline units - 32-channel noise, image embedder removed --
    for i, unit in enumerate(list(pipe.units)):
        if isinstance(unit, WanVideoUnit_NoiseInitializer):
            pipe.units[i] = KinemaDAENoiseInitializer()
            print(f"  units[{i}] NoiseInitializer → KinemaDAE (latent_ch={_lat_ch})")
    _before = len(pipe.units)
    pipe.units = [u for u in pipe.units if not isinstance(u, WanVideoUnit_ImageEmbedderVAE)]
    if len(pipe.units) != _before:
        print(f"  removed ImageEmbedderVAE ({_before} -> {len(pipe.units)}) - t2v has no input image")

    # ── 5) DiT: patchify(32ch) + head + LoRA + ckpt ──
    #   use_async=False means t2_projection is never built and the two checkpoint keys are dropped on purpose
    print("\n[load] DiT surgery + LoRA + checkpoint")
    # Passing --async_delta takes the async path, which builds t2_projection.
    _use_async = (args.async_delta is not None)
    setup_davae_dit(
        pipe, args.dit_checkpoint,
        vae_z_dim=args.vae_z_dim, vae_prior_z_dim=args.vae_prior_z_dim,
        lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank,
        use_dual_schedule=False, use_async=_use_async,
    )
    if not args.no_merge_lora:
        fold_lora(pipe.dit)   # default. It is what the paper's latency assumes, and bf16 rounding makes it not bit-exact
    _pe = pipe.dit.patch_embedding.weight.shape
    print(f"  patch_embedding.weight={tuple(_pe)}")
    assert _pe[1] == args.vae_z_dim + args.vae_prior_z_dim, \
        f"[FAIL] patchify takes {_pe[1]}ch - t2v needs 32 (72 means i2v settings leaked in)"
    if _use_async:
        assert hasattr(pipe.dit, "t2_projection"), \
            "[FAIL] --async_delta was given but the model has no t2_projection"
        # Check the checkpoint really was trained with async. On a pure checkpoint |w|=0, so passing
        # delta would not change the trunk conditioning at all - it would only look as if it did.
        _w = pipe.dit.t2_projection[1].weight
        _wm = float(_w.abs().max())
        assert _wm > 0.0, (
            "[FAIL] t2_projection weights are all zero - this checkpoint was trained without the delta path. "
            "Drop --async_delta and run the single-timestep path")
        print(f"  t2_projection |w|max={_wm:.4e} (async training confirmed)")
    else:
        assert not hasattr(pipe.dit, "t2_projection"), \
            "[FAIL] t2_projection exists although use_async=False - the dual-timestep path is on"

    if args.rope_pos_scale:
        pipe.dit.rope_pos_scale = tuple(float(v) for v in args.rope_pos_scale.split(","))
        print(f"  rope_pos_scale={pipe.dit.rope_pos_scale}")

    t_load = time.time() - t_start
    print(f"\n[TIMING] load {t_load:.1f}s")

    # -- 6) sampling ------------------------------------------------------
    done = 0
    t_gen = time.time()
    _only = None
    if args.only_idx.strip():
        _only = {int(x) for x in args.only_idx.replace(" ", "").split(",") if x}
        print(f"[only_idx] generating only {sorted(_only)} (the plan is unchanged, so idx<->prompt still holds)")
    for d in dims:
        for ent in manifest[d]:
            if _only is not None and ent["idx"] not in _only:
                continue
            out_path = os.path.join(args.out_dir, ent["path"])
            if os.path.exists(out_path):
                print(f"[{d}] {ent['idx']:02d} [skip: exists]"); continue
            print(f"\n[{d}] {ent['idx']:02d}/{len(manifest[d])-1}  {ent['prompt'][:90]}")
            t0 = time.time()
            with torch.no_grad():
                if _use_async:
                    frames = dual_schedule_generate_t2v(
                        pipe, ent["prompt"], NEG_PROMPT,
                        height=args.height, width=args.width, num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        seed=args.seed, shift_main=args.shift_main, z_dim=args.vae_z_dim,
                        async_delta=args.async_delta, cfg_scale=args.cfg_scale,
                        async_shift_prior=args.async_shift_prior, tiled=True)
                else:
                    frames = pipe(                       # * no input_image, no timestep2
                        prompt=ent["prompt"],
                        negative_prompt=NEG_PROMPT,
                        height=args.height, width=args.width, num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        cfg_scale=args.cfg_scale,
                        seed=args.seed, tiled=True,
                    )
            _save(frames, out_path, fps=args.fps)   # default quality=9 - the same as the V-RAE gFVD protocol
            #   (evaluation/common/runtime.py:_atomic_mp4 - imageio quality=9). Real and fake must go through the
            #   same codec path so H.264 degradation cancels in FVD. The protocol locks this format as a constant.
            if args.save_latents and _cap["latents"] is not None:
                # decode_prior_only.py replaces '_latents.pt' with '_prioronly.mp4', so the suffix is fixed.
                _lp = out_path[:-4] + "_latents.pt"
                torch.save(_cap["latents"], _lp)
                print(f"  latents {tuple(_cap['latents'].shape)} → {_lp}")
                _cap["latents"] = None
            done += 1
            print(f"  saved {out_path}  ({time.time()-t0:.1f}s)", flush=True)

    dt = time.time() - t_gen
    print(f"\n[TIMING] {done} videos in {dt:.1f}s ({dt/max(done,1):.1f}s/video), total {time.time()-t_start:.1f}s")
    print(f"Done! {args.out_dir}")


if __name__ == "__main__":
    main()
