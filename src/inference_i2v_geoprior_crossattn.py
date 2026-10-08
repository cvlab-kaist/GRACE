"""
Geoprior DiT I2V inference script.
Loads Geoprior VAE + LoRA DiT (with patchify/head from checkpoint).

Usage:
    # Single GPU
    python dit_training/inference_i2v_geoprior.py \
        --dit_checkpoint results/dit/dit_i2v_v2_caption_freeze_88k_nopatchify/step-5250.safetensors \
        --vae_checkpoint results/decoder_only_freeze88k-lr8.00e-05-bs8-rs256-sr2-fr17/checkpoint-126000.ckpt \
        --image_dir /data/.../vbench_i2v_images/crop/3-2 \
        --prompts_json /data/.../vbench2_i2v_full_info.json \
        --output_dir results/dit/inference/i2v_geoprior \
        --device cuda:0

    # Multi-GPU (split by index range)
    python dit_training/inference_i2v_geoprior.py ... --device cuda:0 --start_idx 0 --n_videos 45
    python dit_training/inference_i2v_geoprior.py ... --device cuda:1 --start_idx 45 --n_videos 45
    ...
"""
import os, sys, argparse, json, types, torch, re
import numpy as np
from PIL import Image
import imageio

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
# [release] Wan2.1-I2V-14B-480P base weights (7 diffusion shards + google/umt5-xxl tokenizer).
#   Point $GRACE_WAN_I2V_DIR at your download; the old default assumed a checkpoints/ symlink
#   that only exists inside the research tree, so a clean clone crashed on it.
_WAN_I2V = os.environ.get("GRACE_WAN_I2V_DIR", os.path.join(_ROOT, "checkpoints/Wan2.1-I2V-14B-480P"))
# [release] The DiffSynth fork this model needs ships at third_party/DiffSynth-Studio.
#   The old default pointed at a sibling of this repo, which does not exist in a clean clone,
#   so the sys.path insert was a no-op and the import below failed outright
#   (or, worse, picked up an unrelated copy without rope_pos_scale). Override with $DIFFSYNTH_ROOT.
os.environ.setdefault(
    "DIFFSYNTH_ROOT",
    os.path.join(_ROOT, "third_party", "DiffSynth-Studio"),
)
_DIFFSYNTH = os.environ["DIFFSYNTH_ROOT"]
# GRACE-style modules ship under modules/ in this repo, but a sibling
# GRACE checkout can be used via $GRACE_ROOT for full module compatibility.
_GRACE_ROOT = os.environ.get("GRACE_ROOT", os.path.join(_ROOT, "modules"))
for p in [_ROOT, _THIS, _DIFFSYNTH, _GRACE_ROOT]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model
from einops import rearrange

# rope_pos_scale consumption guard - same as the t2v version (inference_t2v_geoprior.py:78-89).
#   Without it, a DiffSynth copy that has no rope_pos_scale support can be loaded and the
#   `pipe.dit.rope_pos_scale = _rps` assignment at :1047 is silently ignored, sampling with plain rope.
#   The log still prints '[rope-scale] dit.rope_pos_scale=(2,2,2)', which makes it worse.
#   This actually happened: 139 i2v videos were generated invalid this way.
import diffsynth.pipelines.wan_video as _wv_chk   # noqa: E402
_WV_SRC = os.path.abspath(_wv_chk.__file__)
with open(_WV_SRC, encoding="utf-8") as _f:
    _WV_HAS_ROPE = "rope_pos_scale" in _f.read()
print(f"[diffsynth] {_WV_SRC}\n            rope_pos_scale applied={_WV_HAS_ROPE}", flush=True)
if not _WV_HAS_ROPE:
    raise SystemExit(
        "[FAIL] The DiffSynth copy that was loaded has no rope_pos_scale support, so\n"
        "       --rope_pos_scale is silently ignored and the latents will not match\n"
        "       training. Point DIFFSYNTH_ROOT at third_party/DiffSynth-Studio in this repo.\n"
        f"       Loaded: {_WV_SRC}")
from diffsynth.pipelines.wan_video import (
    WanVideoPipeline, ModelConfig,
    WanVideoUnit_NoiseInitializer, WanVideoUnit_ImageEmbedderVAE,
)
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.models.wan_video_dit import Head as WanHead
from diffsynth.utils.data import save_video
from grace_video_vae import load_grace_geoprior_vae
# [crossattn] VAE loader compatible with first-frame cross-attention (used to swap only the decoder)
from grace_video_vae_crossattn import load_grace_geoprior_vae_crossattn

# dual_schedule (per-branch timestep shift) - only active with the --dual_schedule flag.
# When off, the symbols below are imported but never used, so behaviour is unchanged.
from dual_sched_core import wan_sigmas, dual_step, async_ladders, async_ladders_decoupled
from davae_head_dual import DaVaeHeadDual
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


# --- Copied from train_dit.py to avoid import chain dependencies ---

class DaVaeHead(torch.nn.Module):
    def __init__(self, head_main, head_prior):
        super().__init__()
        self.head_main = head_main
        self.head_prior = head_prior

    def forward(self, x, mod):
        return torch.cat([self.head_main(x, mod), self.head_prior(x, mod)], dim=-1)


class GRACENoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"),
            output_params=("noise",)
        )

    def process(self, pipe, height, width, num_frames, seed, rand_device, vace_reference_image):
        tf = getattr(pipe.vae, 'temporal_factor', 4)
        length = (num_frames - 1) // tf + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        latent_ch = getattr(pipe.vae, 'latent_channels', getattr(pipe.vae, 'z_dim', pipe.vae.model.z_dim))
        shape = (1, latent_ch, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        # Time-correlated initial noise: noise_t = a*shared + sqrt(1-a^2)*iid_t
        #   Reduces the latent flicker that per-frame independent sampling produces (a=0 is the old path).
        _a = float(getattr(pipe, "_init_noise_alpha", 0.0) or 0.0)
        if _a > 0.0:
            _sh2 = (shape[0], shape[1], shape[2] + 1, shape[3], shape[4])
            _n = pipe.generate_noise(_sh2, seed=seed, rand_device=rand_device)
            _shared, _iid = _n[:, :, -1:], _n[:, :, :shape[2]]
            noise = _a * _shared + (1.0 - _a * _a) ** 0.5 * _iid
        else:
            noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}


class GRACEImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y", "ff_static_latent"),
            onload_model_names=("vae",)
        )

    def process(self, pipe, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        sf = pipe.vae.upsampling_factor
        tf = getattr(pipe.vae, 'temporal_factor', 4)
        lat_h, lat_w = height // sf, width // sf

        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames - 2, height, width).to(image.device), end_image.transpose(0, 1)], dim=1)
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)], dim=1)

        msk = torch.ones(1, num_frames, lat_h, lat_w, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            msk[:, -1:] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=tf, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // tf, tf, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]  # (tf, T_lat, H', W')

        # encode stays eval-chunked, matching stage-2 training (train_dit keeps vae.eval() throughout).
        y_full = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y_full = y_full.to(dtype=pipe.torch_dtype, device=pipe.device)

        y = torch.concat([msk, y_full])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        # [FrameInit static-encode] Encode a real static video made by repeating the first frame num_frames times.
        #   repeat(z0) has phase and statistics error in the later latent slices because of the 1+8(T-1) layout;
        #   this path is exact. Wan's causal VAE leaves the first slice identical and only replaces the rest.
        ff_static_latent = None
        if getattr(pipe, '_frameinit_static_encode', False):
            static_input = image.transpose(0, 1).repeat(1, num_frames, 1, 1)   # (3, num_frames, H, W) static video
            ff_static_latent = pipe.vae.encode([static_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
                                               device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            ff_static_latent = ff_static_latent.to(dtype=pipe.torch_dtype, device=pipe.device).unsqueeze(0)  # (1, 2z, T_lat, H', W')
        return {"y": y, "ff_static_latent": ff_static_latent}

NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def build_model_paths():
    ckpt_dir = _WAN_I2V
    return [
        [os.path.join(ckpt_dir, f"diffusion_pytorch_model-0000{i}-of-00007.safetensors")
         for i in range(1, 8)],
        os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(ckpt_dir, "Wan2.1_VAE.pth"),
        os.path.join(ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]

# GRACE_LOG_INFO=1 enables INFO logging.
#   The decoder-swap overlay key count, the ff_inject load state and other diagnostics all go through
#   logging.info, so at the default root=WARNING they are invisible and nothing confirms they applied.
import logging as _lg_early, os as _os_early   # this block can run before the module-level imports, so it imports what it needs
if _os_early.environ.get("GRACE_LOG_INFO"):
    _lg_early.basicConfig(level=_lg_early.INFO, force=True)


# [release] One place decides how videos are written. Default matches the paper runs.
_LOSSLESS = False   # bit-exact RGB (libx264rgb). FOR METRICS ONLY - gbrp renders green in players.
_MAXQ = True        # default: libx264 crf 0, yuv420p - visually lossless and plays everywhere


def _save(frames, out_path, fps=16):
    if _MAXQ:
        import imageio as _iio, numpy as _np
        w = _iio.get_writer(out_path, fps=fps, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=1, output_params=["-crf", "0", "-preset", "veryslow"])
        for f in frames:
            w.append_data(_np.asarray(f))
        w.close()
    elif _LOSSLESS:
        import imageio as _iio, numpy as _np
        w = _iio.get_writer(out_path, fps=fps, codec="libx264rgb", pixelformat="rgb24",
                            macro_block_size=1, output_params=["-crf", "0", "-preset", "veryslow"])
        for f in frames:
            w.append_data(_np.asarray(f))
        w.close()
    else:
        save_video(frames, out_path, fps=fps, quality=9)


def fold_lora(dit):
    """Fold the LoRA deltas into the base weights and drop the wrappers.

    Mathematically a no-op, but it is how the released model is meant to run and how the
    paper's latency was measured: with the wrappers in place every forward pays an extra
    (x@A)@B plus a Python call through peft's lora.Linear. merge() alone keeps that call,
    so the parent's attribute is swapped back to the plain nn.Linear as well.
    """
    from peft.tuners.lora import LoraLayer
    n = 0
    for m in dit.modules():
        if isinstance(m, LoraLayer):
            m.merge(); n += 1
    swapped = 0
    for parent in dit.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoraLayer):
                setattr(parent, name, child.base_layer); swapped += 1
    left = sum(1 for m in dit.modules() if isinstance(m, LoraLayer))
    assert left == 0, f"[FAIL] {left} LoRA wrappers left after folding"
    print(f"  [lora] folded {n} layers, unwrapped {swapped}")
    return n


def setup_davae_dit(pipe, dit_checkpoint_path, vae_z_dim=32, vae_prior_z_dim=16,
                    lora_target_modules="q,k,v,o,k_img,v_img,ffn.0,ffn.2", lora_rank=512,  # k_img and v_img are no longer missing from the default (it now matches argparse)
                    mask_ch=None, wan_mask_ch=4, use_dual_schedule=False, use_async=False):
    """Set up DA-VAE patchify/head + LoRA on DiT, then load checkpoint.

    Args:
        vae_z_dim: z_main channels (e.g. 32).
        vae_prior_z_dim: z_prior channels (e.g. 16); must match Wan native VAE z_dim
            since pretrained patch_embedding weights are sliced as 1:1 prior copies.
        mask_ch: number of mask channels in I2V `y` tensor (= VAE temporal_factor).
            If None, taken from `pipe.vae.temporal_factor` (default 8 for geoprior).
        wan_mask_ch: Wan native I2V mask channels (default 4 = native temporal_factor).
    """
    dit = pipe.dit
    dim = dit.dim
    patch_size = dit.patch_size
    eps = dit.head.norm.eps
    _dev = next(dit.parameters()).device
    _dtype = next(dit.parameters()).dtype
    _has_image = getattr(dit, 'has_image_input', False)
    _main_ch = vae_z_dim
    _prior_ch = vae_prior_z_dim
    # Mask channels in our `y` tensor = VAE temporal_factor (e.g. 8 for geoprior).
    if mask_ch is None:
        mask_ch = int(getattr(getattr(pipe, 'vae', None), 'temporal_factor', 8))
    _mask_ch = mask_ch
    # Wan I2V pretrained patch_embedding layout assumption:
    #   [noisy(0:_WAN_NOISY_CH) | mask(:+_WAN_MASK_CH) | image(:+_WAN_NOISY_CH)]
    # with Wan native VAE z_dim == _prior_ch (16) and Wan mask == 4 (temporal_factor=4).
    _WAN_MASK_CH = int(wan_mask_ch)
    _WAN_NOISY_CH = _prior_ch
    # Sanity: we can only copy min(_mask_ch, _WAN_MASK_CH) mask channels from pretrained.
    _mask_copy_ch = min(_mask_ch, _WAN_MASK_CH)

    # Save original patch_embedding weights
    _orig_pe_w = dit.patch_embedding.weight.data.clone()
    _orig_pe_b = dit.patch_embedding.bias.data.clone()

    # Our patchify input layout:
    #   [noisy_z_main | noisy_z_prior | mask | image_z_main | image_z_prior]
    # Offsets:
    _noisy_main_off = 0
    _noisy_prior_off = _main_ch
    _mask_off = _main_ch + _prior_ch
    _img_main_off = _main_ch + _prior_ch + _mask_ch
    _img_prior_off = _main_ch + _prior_ch + _mask_ch + _main_ch

    # Total input channels
    if _has_image:
        _patchify_in_ch = (_main_ch + _prior_ch) * 2 + _mask_ch
    else:
        _patchify_in_ch = _main_ch + _prior_ch
    pe = torch.nn.Conv3d(_patchify_in_ch, dim, kernel_size=patch_size, stride=patch_size)
    with torch.no_grad():
        # z_main slices are left zero (no pretrained counterpart); zero_() handles this.
        pe.weight.zero_()
        pe.bias.copy_(_orig_pe_b)
        # noisy_z_prior ← pretrained noisy (Wan noisy occupies [0:_WAN_NOISY_CH])
        pe.weight[:, _noisy_prior_off : _noisy_prior_off + _prior_ch] = (
            _orig_pe_w[:, 0:_WAN_NOISY_CH]
        )
        if _has_image:
            # mask ← pretrained mask (copy min(mask_ch, _WAN_MASK_CH); rest zero)
            pe.weight[:, _mask_off : _mask_off + _mask_copy_ch] = (
                _orig_pe_w[:, _WAN_NOISY_CH : _WAN_NOISY_CH + _mask_copy_ch]
            )
            # image_z_prior ← pretrained image
            pe.weight[:, _img_prior_off : _img_prior_off + _prior_ch] = (
                _orig_pe_w[:, _WAN_NOISY_CH + _WAN_MASK_CH : _WAN_NOISY_CH + _WAN_MASK_CH + _prior_ch]
            )
    dit.patch_embedding = pe.to(device=_dev, dtype=_dtype)

    # Replace head: DaVaeHead
    head_prior = WanHead(dim, _prior_ch, patch_size, eps)
    head_prior.norm.load_state_dict(dit.head.norm.state_dict())
    head_prior.head.weight.data.copy_(dit.head.head.weight.data)
    head_prior.head.bias.data.copy_(dit.head.head.bias.data)
    head_prior.modulation.data.copy_(dit.head.modulation.data)

    head_main = WanHead(dim, _main_ch, patch_size, eps)
    head_main.norm.load_state_dict(dit.head.norm.state_dict())
    head_main.modulation.data.copy_(dit.head.modulation.data)
    torch.nn.init.zeros_(head_main.head.weight)
    torch.nn.init.zeros_(head_main.head.bias)

    # With dual_schedule, head_prior takes its own t_prior embedding, so DaVaeHeadDual is used.
    # With t_prior_emb=None (the default) it is mathematically identical to DaVaeHead, so off is lossless.
    if use_dual_schedule or use_async:
        # async also needs t_prior_emb, hence DaVaeHeadDual (identical to DaVaeHead when t_prior_emb=None)
        dit.head = DaVaeHeadDual(head_main, head_prior).to(device=_dev, dtype=_dtype)
    else:
        dit.head = DaVaeHead(head_main, head_prior).to(device=_dev, dtype=_dtype)

    # Monkey-patch patchify
    def _da_vae_patchify(self, x, control_camera_latents_input=None, enable_wantodance_global=False):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x
    dit.patchify = types.MethodType(_da_vae_patchify, dit)

    # Monkey-patch unpatchify
    _px, _py, _pz = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    _main_flat = _main_ch * _px * _py * _pz
    _prior_flat = _prior_ch * _px * _py * _pz

    def _da_vae_unpatchify(self, x, grid_size):
        x_main = x[..., :_main_flat]
        x_prior = x[..., _main_flat:]
        f, h, w = grid_size
        out_main = rearrange(x_main, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
                             f=f, h=h, w=w, x=_px, y=_py, z=_pz)
        out_prior = rearrange(x_prior, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
                              f=f, h=h, w=w, x=_px, y=_py, z=_pz)
        return torch.cat([out_main, out_prior], dim=1)
    dit.unpatchify = types.MethodType(_da_vae_unpatchify, dit)

    # Inject LoRA
    target_modules = lora_target_modules.split(",")
    if len(target_modules) == 1:
        target_modules = target_modules[0]
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=target_modules)
    dit = inject_adapter_in_model(lora_config, dit)

    # [async] t2_projection is built only when use_async is set (same structure as training, zero-init).
    # It must exist before loading so the t2 keys of an async checkpoint are absorbed by load_state_dict below.
    # An older checkpoint leaves it zero, so timestep2 becomes a no-op. With the flag off the module is absent.
    if use_async:
        dit.t2_projection = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(dim, dim * 6))
        torch.nn.init.zeros_(dit.t2_projection[1].weight)
        torch.nn.init.zeros_(dit.t2_projection[1].bias)
        dit.t2_projection.to(device=_dev, dtype=_dtype)

    pipe.dit = dit

    # Load checkpoint: patch_embedding + head + LoRA weights
    state_dict = load_file(dit_checkpoint_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A.weight" in key or "lora_B.weight" in key:
            new_key = key.replace("lora_A.weight", "lora_A.default.weight") \
                         .replace("lora_B.weight", "lora_B.default.weight")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    missing, unexpected = pipe.dit.load_state_dict(new_state_dict, strict=False)
    print(f"DiT checkpoint loaded: {len(new_state_dict)} keys, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    # unexpected>0 means trained keys from the checkpoint were silently dropped (usually a
    # lora_target_modules mismatch). t2_projection is excluded: it is dropped on purpose when use_async=False.
    _dropped = [k for k in unexpected if "t2_projection" not in k]
    if _dropped:
        print(f"  WARNING: {len(_dropped)} trained keys could not be loaded and were dropped. "
              f"Check that lora_target_modules matches training. First few: {_dropped[:3]}")

    # Verify critical keys loaded
    pe_loaded = any('patch_embedding' in k for k in new_state_dict)
    head_loaded = any('head' in k for k in new_state_dict)
    lora_loaded = any('lora' in k for k in new_state_dict)
    print(f"  patch_embedding: {'OK' if pe_loaded else 'MISSING!'}")
    print(f"  head (main+prior): {'OK' if head_loaded else 'MISSING!'}")
    print(f"  LoRA: {'OK' if lora_loaded else 'MISSING!'}")
    if use_async:
        _t2_in_ckpt = any('t2_projection' in k for k in new_state_dict)
        print(f"  t2_projection: {'loaded (async ckpt)' if _t2_in_ckpt else 'zero-init (old ckpt, timestep2 is a no-op)'}")

    pipe.dit.to(device=_dev, dtype=_dtype)

    # Replace NoiseInitializer and ImageEmbedderVAE
    for i, unit in enumerate(pipe.units):
        if isinstance(unit, WanVideoUnit_NoiseInitializer):
            pipe.units[i] = GRACENoiseInitializer()
            print(f"Replaced NoiseInitializer at units[{i}]")
            if False:
                print(f"[init-noise] temporal-correlated alpha={pipe._init_noise_alpha}")
        elif isinstance(unit, WanVideoUnit_ImageEmbedderVAE):
            pipe.units[i] = GRACEImageEmbedderVAE()
            print(f"Replaced ImageEmbedderVAE at units[{i}]")


# dual_schedule sampling loop. It replaces the denoise loop inside pipe(); input preparation
# (text, CLIP, image y, noise) is reused through pipe.unit_runner and only the denoise step runs
# z_main and z_prior on their own sigma ladders (shift_main / shift_prior). wan_video.py is untouched.
# Only head_prior receives the t_prior embedding (dit.head.t_prior_emb); trunk and head_main use t_main.
# With shift_main == shift_prior this equals the single-schedule pipe(), which guarantees no regression.
@torch.no_grad()
def dual_schedule_generate(pipe, prompt, negative_prompt, input_image,
                           height, width, num_frames, num_inference_steps, seed,
                           shift_main, shift_prior, z_dim, tiled=True,
                           tile_size=(30, 52), tile_stride=(15, 26), cfg_scale=5.0,
                           vscale_main=1.0, async_delta=None, first_frame_inject=False,
                           async_reverse=False,
                           async_shift_prior=None, async_decoupled=None, first_frame_inject_clean=False,
                           prior_floor_cond=None,
                           cfg_main=None, restart_sigma=None, restart_steps=20,
                           prior_cfg_scale=None, prior_cfg_null_y=False, prior_cfg_weak_delta=None,
                           prior_cfg_main_only=False, inject_prior_latents=None,
                           use_frameinit=False, frameinit_sigma=0.9, frameinit_d_s=0.25,
                           frameinit_d_t=0.25, frameinit_filter='gaussian', frameinit_target='all'):
    dev, dt = pipe.device, pipe.torch_dtype
    # [FrameInit] values passed as pipe attributes win (args -> pipe -> here, without touching call sites).
    use_frameinit = getattr(pipe, '_use_frameinit', use_frameinit)
    frameinit_sigma = getattr(pipe, '_frameinit_sigma', frameinit_sigma)
    frameinit_d_s = getattr(pipe, '_frameinit_d_s', frameinit_d_s)
    frameinit_d_t = getattr(pipe, '_frameinit_d_t', frameinit_d_t)
    frameinit_filter = getattr(pipe, '_frameinit_filter', frameinit_filter)
    frameinit_target = getattr(pipe, '_frameinit_target', frameinit_target)
    frameinit_truncstart = getattr(pipe, '_frameinit_truncstart', False)
    # [truncated start] With frameinit and truncstart, the ladder starts at sigma0 = frameinit_sigma, not 1.0.
    #   The declared noise level sigma0 then equals the actual signal content (1-sigma0) of the initial x,
    _fi_smax = float(frameinit_sigma) if (use_frameinit and frameinit_truncstart) else 1.0
    def _wan_sigmas_trunc(N, sh, smax):
        # Same grid as wan_sigmas (u=linspace(u_top,0,N+1)[:-1]) but starting at the sigma0 bound u_top=warp^-1(sigma0).
        u_top = float(smax) / (sh - (sh - 1.0) * float(smax))       # smax=1.0 gives u_top=1.0, the old behaviour
        u = torch.linspace(u_top, 0.0, N + 1)[:-1]
        return sh * u / (1.0 + (sh - 1.0) * u)
    # set_timesteps once so units that read pipe.scheduler behave; use t_main as the trunk clock
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=shift_main)

    inputs_posi = {
        "prompt": prompt,
        "vap_prompt": None,
        "tea_cache_l1_thresh": None, "tea_cache_model_id": "", "num_inference_steps": num_inference_steps,
    }
    inputs_nega = {
        "negative_prompt": negative_prompt,
        "negative_vap_prompt": None,
        "tea_cache_l1_thresh": None, "tea_cache_model_id": "", "num_inference_steps": num_inference_steps,
    }
    inputs_shared = {
        "input_image": input_image,
        "end_image": None,
        "input_video": None, "denoising_strength": 1.0,
        "control_video": None, "reference_image": None,
        "camera_control_direction": None, "camera_control_speed": 1/54, "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None, "vace_video_mask": None, "vace_reference_image": None, "vace_scale": 1.0,
        "seed": seed, "rand_device": "cpu",
        "height": height, "width": width, "num_frames": num_frames,
        "cfg_scale": cfg_scale, "cfg_merge": False,
        "sigma_shift": shift_main,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        "sliding_window_size": None, "sliding_window_stride": None,
        "input_audio": None, "audio_sample_rate": 16000, "s2v_pose_video": None, "audio_embeds": None, "s2v_pose_latents": None, "motion_video": None,
        "animate_pose_video": None, "animate_face_video": None, "animate_inpaint_video": None, "animate_mask_video": None,
        "vap_video": None,
        "wantodance_music_path": None, "wantodance_reference_image": None, "wantodance_fps": 30,
        "wantodance_keyframes": None, "wantodance_keyframes_mask": None,
        "framewise_decoding": False,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    # two sigma ladders; both length = num_inference_steps
    if async_decoupled is not None:
        # decoupled: prior runs the full [1,0] ladder, main is offset to [1+offset,0]. timestep2 is passed.
        sig_main, sig_prior = async_ladders_decoupled(num_inference_steps, shift_main, async_decoupled)
        sig_main, sig_prior = sig_main.to(dev), sig_prior.to(dev)
        _pass_t2 = True
    elif async_delta is not None:
        # [async] offset ladder, same geometry as training: a constant delta in u space, phases at both ends, a floor.
        # The conditioning t = sigma*1000, so phase 3 (sigma=0.02, t=20) is a pair the model saw in training.
        # Passing async_shift_prior as well keeps the offset and warps only the base ladder with a different shift.
        #   With async_shift_prior=None both branches share a shift, which is the plain offset-only case.
        _psf = float(getattr(pipe, '_prior_sigma_floor', 0.02))
        sig_main, sig_prior = async_ladders(num_inference_steps, shift_main, async_delta,
                                            shift_prior=async_shift_prior, sigma_max=_fi_smax,
                                            sigma_min=_psf,
                                            prior_tail_knee=getattr(pipe, '_prior_tail_knee', None))
        # [REV] Inference mirror of the reversed ablation (trained with delta<0, so z_main is cleaner than z_prior).
        #   dual_sched_core.async_ladders only supports delta>0 (dmax=clamp(min=0)), so a negative delta would
        #   neither stretch the ladder nor apply the floor. The exact mirror is to swap the two ladders wholesale,
        #   which moves the lead and the sigma_min floor to main together.
        if async_reverse:
            sig_main, sig_prior = sig_prior, sig_main
        # The sigma_min in async_ladders applies only in the saturated region, so a larger floor (0.05+) creates a
        #   non-monotone stretch where terms dip below the floor and come back. A global clamp guarantees
        #   'monotone descent, reach the floor, freeze'. At the default floor=0.02 a 50-step grid skips the
        #   (0, 0.02) interval anyway, so this is equivalent to the old behaviour.
        if _psf > 0.0:
            sig_prior = sig_prior.clamp(min=_psf)
        sig_main, sig_prior = sig_main.to(dev), sig_prior.to(dev)
        _pass_t2 = True
    elif async_shift_prior is not None:
        # shift-based dual (no offset): two ladders, shift_main (=5) and async_shift_prior.
        #   Unlike the offset form there is no phase freeze or saturation, so each branch uses all 50 steps.
        sig_main = _wan_sigmas_trunc(num_inference_steps, shift_main, _fi_smax).to(dev)
        sig_prior = _wan_sigmas_trunc(num_inference_steps, async_shift_prior, _fi_smax).to(dev)
        _pass_t2 = True
    else:
        sig_main = _wan_sigmas_trunc(num_inference_steps, shift_main, _fi_smax).to(dev)
        sig_prior = _wan_sigmas_trunc(num_inference_steps, shift_prior, _fi_smax).to(dev)
        _pass_t2 = False
    ts_main = (sig_main * 1000.0)
    ts_prior = (sig_prior * 1000.0)
    # [opt-in] Force the base conditioning value inside the floor region. The ladder sigmas and the Euler
    #   steps are unchanged; only the declared value is replaced, for checkpoints trained before the fix.
    if prior_floor_cond is not None:
        _fl = float(getattr(pipe, '_prior_sigma_floor', 0.02))
        _at_floor = (sig_prior <= _fl + 1e-6)
        ts_prior = torch.where(_at_floor, torch.full_like(ts_prior, float(prior_floor_cond)), ts_prior)
        print(f"[prior_floor_cond] forcing the base conditioning on {int(_at_floor.sum())}/{len(sig_prior)} steps "
              f"at floor({_fl}) to {prior_floor_cond} (default would be {_fl*1000:.1f})", flush=True)

    # [FrameInit / ConsistI2V] Re-initialize the noise: low frequencies from the first-frame latent, high from noise.
    #   z_sigma = (1-sigma)*ff_static + sigma*noise (flow matching), with freq_mix taking only the low band.
    #   Both live in the same space: inputs_shared['latents'] is the normalized DiT noise and the clean
    #   first-frame latent in _y is normalized the same way.
    if use_frameinit:
        from frameinit_utils import get_freq_filter, freq_mix_3d
        _lat = inputs_shared["latents"]                          # (1, 2z, T, H, W) init noise
        _yfi = inputs_shared["y"]
        _mchfi = _yfi.shape[1] - _lat.shape[1]                   # number of mask channels (= tf)
        _ff0 = _yfi[:, _mchfi:, 0:1].clone()                     # (1, 2z, 1, H, W) clean first-frame latent
        # frameinit_norm_ff: normalize the first-frame latent to zero mean and unit std per channel,
        #   matching the ConsistI2V convention of vae.encode * scaling_factor (our _y latents are std 0.34-0.59).
        if getattr(pipe, '_frameinit_norm_ff', False):
            _m = _ff0.mean(dim=(3, 4), keepdim=True)
            _s = _ff0.std(dim=(3, 4), keepdim=True).clamp(min=1e-4)
            _ff0 = (_ff0 - _m) / _s
        # Prefer the static encode: use the real static-video encoding as the anchor, else fall back to repeat(z0).
        _ff_static_enc = inputs_shared.get("ff_static_latent", None)
        if _ff_static_enc is not None:
            _ff_static = _ff_static_enc.to(_lat)                 # (1,2z,T_lat,H',W') correct encoding of the static video
        else:
            _ff_static = _ff0.repeat(1, 1, _lat.shape[2], 1, 1)  # repeating along T approximates a static first-frame video
        _sg = float(frameinit_sigma)
        _lpf = get_freq_filter(_lat.shape, dev, frameinit_filter, 4, frameinit_d_s, frameinit_d_t)
        if frameinit_truncstart:
            # [truncated start] The ladder starts at sigma0 (see _wan_sigmas_trunc / async_ladders sigma_max).
            #   x0_proxy = LPF(ff_static), so z_init = (1-sigma0)*x0_proxy + sigma0*noise is exactly x_{sigma0}.
            #   The declared sigma0 and the signal content (1-sigma0) agree by construction, which removes the
            #   brightness blowout structurally. This is the plain flow-matching form - no vp coefficients needed.
            _x0proxy = freq_mix_3d(_ff_static.float(),
                                   torch.zeros_like(_ff_static).float(), LPF=_lpf).to(_lat.dtype)  # = LPF(ff)
            # With truncstart the noise of **all** channels must be scaled by sigma0, regardless of the target,
            # or the ladder's declared sigma0 is wrong. Previously channels outside the target kept unit variance
            # (for target=prior, z_main carried 11% excess noise), which is where the residual speckle came from.
            _lat = _sg * _lat                                    # all channels: the noise component of x_{sigma0}
            inputs_shared["latents"] = _lat
            _mixed = _lat + (1.0 - _sg) * _x0proxy               # the anchor is added only on the target channels (sliced below)
        else:
            # [kept for reproducibility] variance-preserving mix + freq_mix. It disagrees with the schedule and
            _a = 1.0 - _sg                                       #   shows brightness blowout.
            _b = (1.0 - _a * _a) ** 0.5                          # variance-preserving noise amplitude
            _z_sg = _a * _ff_static.to(_lat) + _b * _lat         # at sigma=1 this is pure noise (a=0, b=1)
            _mixed = freq_mix_3d(_z_sg.float(), _lat.float(), LPF=_lpf).to(_lat.dtype)
        # frameinit_skip_frame0: leave the frame-0 init alone and apply this only to frames 1 and later.
        #   ConsistI2V removes frame 0 from the denoising latent entirely, so the static anchor only affects the rest.
        _tsl = slice(1, None) if getattr(pipe, '_frameinit_skip_frame0', False) else slice(None)
        if frameinit_target == 'prior':
            _lat[:, z_dim:, _tsl] = _mixed[:, z_dim:, _tsl]      # only the 16 z_prior channels (the low-frequency structure branch)
        elif frameinit_target == 'main':
            _lat[:, :z_dim, _tsl] = _mixed[:, :z_dim, _tsl]
        else:                                                    # 'all' - every one of the 32 channels, as in ConsistI2V
            _lat[:, :, _tsl] = _mixed[:, :, _tsl]
        inputs_shared["latents"] = _lat
        print(f"[frameinit] target={frameinit_target} sigma={_sg} d_s={frameinit_d_s} d_t={frameinit_d_t} "
              f"filter={frameinit_filter} truncstart={frameinit_truncstart} (ladder σ0={_fi_smax})")

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    dit = pipe.dit
    # decode offload return: if the previous video moved the DiT to the CPU for decoding, bring it back
    if next(dit.parameters()).device.type != "cuda":
        dit.to(dev)
    # first-frame GT injection (RePaint style): at every step the first latent frame is overwritten with the
    #   clean GT latent forward-noised to the current level, anchoring the first frame to the input.
    #   gt_z0 is the clean latent's first frame from the ImageEmbedder y (= [mask|clean_latent]).
    if first_frame_inject or first_frame_inject_clean:
        _y = inputs_shared["y"]
        _mch = _y.shape[1] - inputs_shared["latents"].shape[1]   # number of mask channels (= tf)
        _gt_z0 = _y[:, _mch:, 0:1].clone()                       # (1, 2z, 1, H, W) clean first-frame latent
        _eps0 = inputs_shared["latents"][:, :, 0:1].clone()      # (1, 2z, 1, H, W) first-frame initial noise
    # wrong-prior injection: teacher-force the whole prior slice from a donor scene's clean prior at every
    #   step (same RePaint form as first_frame_inject, but over the full prior channels).
    #   It tests whether the main stream follows a plausible but wrong base. None is a complete no-op.
    if inject_prior_latents is not None:
        _zp_wrong = inject_prior_latents.to(device=dev, dtype=inputs_shared["latents"].dtype)
        _eps_p = inputs_shared["latents"][:, z_dim:].clone()     # (1, z, T, H, W) fixed initial noise for the base branch
    for i in range(num_inference_steps):
        t_main = ts_main[i:i + 1].to(dtype=dt, device=dev)
        t_prior = ts_prior[i:i + 1].to(dtype=dt, device=dev)
        # condition head_prior on the residual's noise level (trunk + head_main stay on t_main)
        if not _pass_t2:
            # legacy dual_schedule path: set the head manually, as before
            dit.head.t_prior_emb = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t_prior).to(dt))
        # [async] with async, timestep2 goes to model_fn, which handles both the t2_projection injection
        # and the head relay, so it is not set twice. Both CFG passes use the same (t_main, t_prior) pair,
        # since the CFG subtraction must differ only by the prompt.
        _t2kw = {"timestep2": t_prior} if _pass_t2 else {}
        v_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=t_main, **_t2kw)
        if cfg_scale != 1.0:
            v_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=t_main, **_t2kw)
            v = v_nega + cfg_scale * (v_posi - v_nega)
            # separate CFG for z_main (the residual). Detail channels depend less on the text, so a lower CFG
            #   can reduce artifact amplification. With cfg_main=None this is the old behaviour (= cfg_scale).
            if cfg_main is not None and cfg_main != cfg_scale:
                v[:, :z_dim] = v_nega[:, :z_dim] + cfg_main * (v_posi[:, :z_dim] - v_nega[:, :z_dim])
        else:
            v = v_posi
        # [prior-CFG] zero-null experiment, with no dropout training behind it. The null pass zeroes only the
        # noisy_prior slice and keeps y, so it isolates the contribution of the dynamic base stream.
        # Failure signature: oversaturation, colour drift, structural artifacts scaling with w.
        if prior_cfg_scale is not None and prior_cfg_scale != 1.0:
            sh_null = dict(inputs_shared)
            _lat_null = inputs_shared["latents"].clone()
            _t2kw_null = dict(_t2kw)
            _skip_pcfg = False
            if prior_cfg_weak_delta is not None:
                # [band contrast] The weak pole renoises the base up to roughly weak_delta of lead, so both poles are
                # states the model was trained on - the sound version of the zero-null gamble.
                _s_d = float(sig_prior[i])                                   # current (deep) base sigma
                _s_m = float(sig_main[i])
                _u_m = _s_m / (shift_main - (shift_main - 1.0) * _s_m)       # inverse warp
                _u_w = max(_u_m - float(prior_cfg_weak_delta), 0.0)
                _s_w = max(shift_main * _u_w / (1.0 + (shift_main - 1.0) * _u_w), 0.02)
                if _s_w > _s_d + 1e-4:
                    # exact up-noising: x_w = a*x_d + c*eps,  a=(1-s_w)/(1-s_d), c=sqrt(s_w^2-(s_d*a)^2)
                    _a = (1.0 - _s_w) / (1.0 - _s_d)
                    _c = (max(_s_w ** 2 - (_s_d * _a) ** 2, 0.0)) ** 0.5
                    _lat_null[:, z_dim:] = _a * _lat_null[:, z_dim:] + _c * torch.randn_like(_lat_null[:, z_dim:])
                    if _pass_t2:
                        _t2kw_null = {"timestep2": torch.tensor([_s_w * 1000.0], dtype=dt, device=dev)}
                else:
                    _skip_pcfg = True   # weak pole == strong pole (they converge late), so the contrast is 0 and guidance is skipped this step
            else:
                _lat_null[:, z_dim:] = 0.0   # the original zero-null mode
                # y-null variant: also zero the img_prior slice of y
                if prior_cfg_null_y and sh_null.get("y") is not None:
                    _y_null = sh_null["y"].clone()
                    _mch = _y_null.shape[1] - 2 * z_dim
                    _y_null[:, _mch + z_dim:] = 0.0
                    sh_null["y"] = _y_null
            if not _skip_pcfg:
                sh_null["latents"] = _lat_null
                v_posi_n = pipe.model_fn(**models, **sh_null, **inputs_posi, timestep=t_main, **_t2kw_null)
                if cfg_scale != 1.0:
                    v_nega_n = pipe.model_fn(**models, **sh_null, **inputs_nega, timestep=t_main, **_t2kw_null)
                    v_null = v_nega_n + cfg_scale * (v_posi_n - v_nega_n)
                else:
                    v_null = v_posi_n
                if prior_cfg_main_only:
                    # extrapolate only the main channels (the cfg_main pattern) and leave the base trajectory untouched,
                    # which removes the global-statistics disturbance that caused saturation and colour drift
                    v = v.clone()
                    v[:, :z_dim] = v_null[:, :z_dim] + prior_cfg_scale * (v[:, :z_dim] - v_null[:, :z_dim])
                else:
                    v = v_null + prior_cfg_scale * (v - v_null)
        # z_main velocity scaling: multiply only the z_main prediction by lambda, to diagnose or correct
        # accumulated bias. lambda=1.0 is a no-op. If a sweep reduces the residual it is bias; if not, training.
        if vscale_main != 1.0:
            v = v.clone(); v[:, :z_dim] = v[:, :z_dim] * vscale_main
        inputs_shared["latents"] = dual_step(inputs_shared["latents"], v, i, sig_main, sig_prior, z_dim)
        # After dual_step, overwrite the base slice with the donor base forward-noised to sigma_prior[i+1]
        #   (the last step is sigma=0, so it is exactly the donor). Applied before first_frame_inject, so the
        #   frame-0 anchor wins if both are used.
        if inject_prior_latents is not None:
            _npj = sig_prior[i + 1] if i + 1 < len(sig_prior) else sig_prior.new_zeros(())
            inputs_shared["latents"][:, z_dim:] = (1 - _npj) * _zp_wrong + _npj * _eps_p
        # first-frame injection: after dual_step, overwrite the first frame with the forward-noised GT.
        #   z_main uses sigma_main[i+1] and z_prior uses sigma_prior[i+1]; the last step is sigma=0, so exactly GT.
        if first_frame_inject_clean:
            # clean variant: no noise - every step overwrites the first frame with the clean GT latent
            #   (not the RePaint forward-noise form; frame 0 stays at sigma=0 while the other frames denoise)
            inputs_shared["latents"][:, :, 0:1] = _gt_z0.to(inputs_shared["latents"].dtype)
        elif first_frame_inject:
            _nm = sig_main[i + 1] if i + 1 < len(sig_main) else sig_main.new_zeros(())
            _np = sig_prior[i + 1] if i + 1 < len(sig_prior) else sig_prior.new_zeros(())
            _lat = inputs_shared["latents"]
            _lat[:, :z_dim, 0:1] = (1 - _nm) * _gt_z0[:, :z_dim] + _nm * _eps0[:, :z_dim]
            _lat[:, z_dim:, 0:1] = (1 - _np) * _gt_z0[:, z_dim:] + _np * _eps0[:, z_dim:]
    dit.head.t_prior_emb = None  # reset state

    # [restart] second pass: renoise only the main stream of the finished video to sigma_r, keeping the base
    # near its floor (0.02). That reproduces a 'noised finished video', the input form training saw.
    # It costs restart_steps extra function evaluations.
    if restart_sigma is not None:
        s_r, s_pf = float(restart_sigma), 0.02
        lat = inputs_shared["latents"]
        lat[:, :z_dim] = (1 - s_r) * lat[:, :z_dim] + s_r * torch.randn_like(lat[:, :z_dim])
        lat[:, z_dim:] = (1 - s_pf) * lat[:, z_dim:] + s_pf * torch.randn_like(lat[:, z_dim:])
        u_r = s_r / (shift_main - (shift_main - 1.0) * s_r)          # inverse warp
        m2 = torch.linspace(u_r, 0.0, restart_steps + 1)[:-1]
        sig_m2 = (shift_main * m2 / (1.0 + (shift_main - 1.0) * m2)).to(dev)
        sig_p2 = torch.full_like(sig_m2, s_pf)                        # constant, so dual_step has d=0 (the base is held and only the final jump takes 0.02 to 0)
        ts_m2, ts_p2 = sig_m2 * 1000.0, sig_p2 * 1000.0               #  only the final to_final jump takes it from 0.02 to 0)
        for i in range(restart_steps):
            t_main2 = ts_m2[i:i + 1].to(dtype=dt, device=dev)
            t_prior2 = ts_p2[i:i + 1].to(dtype=dt, device=dev)
            v_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=t_main2, timestep2=t_prior2)
            if cfg_scale != 1.0:
                v_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=t_main2, timestep2=t_prior2)
                v = v_nega + cfg_scale * (v_posi - v_nega)
            else:
                v = v_posi
            inputs_shared["latents"] = dual_step(inputs_shared["latents"], v, i, sig_m2, sig_p2, z_dim)
        dit.head.t_prior_emb = None

    pipe.load_models_to_device(['vae'])
    # Separate the decode peak from the resident DiT (~28 GB) - load_models_to_device is a no-op without
    #   vram_management. [release] By default the DiT stays on the GPU during decode, the state the paper's
    #   latency was measured in. Set DIT_OFFLOAD_DECODE=1 if VRAM is tight and it is moved out and back
    #   (a PCIe round trip, measured at about 29 s per video). The output is identical either way.
    if os.environ.get("DIT_OFFLOAD_DECODE", "0") != "0":
        dit.to("cpu"); torch.cuda.empty_cache()
    # [crossattn] Feed the first-frame pixels ([-1,1], full resolution HxW) into the decoder cross-attention K/V.
    #   preprocess_image is the same one used to build y_full in ImageEmbedding, so it matches encode.
    #   crossattn is only verified single-pass, hence tiled=False (loaded with force_single_pass).
    _ff_px = pipe.preprocess_image(input_image.resize((width, height))).to(dev)  # (1,3,H,W)
    if getattr(pipe.vae, '_use_spatial_tile', False):
        # spatial tiling lowers the GPU peak (per-tile decode with a per-tile first-frame crop)
        video = pipe.vae.decode(inputs_shared["latents"], device=dev, spatial_tiled=True,
                                spatial_tile_size=pipe.vae._spatial_tile_size,
                                spatial_tile_stride=pipe.vae._spatial_tile_stride,
                                first_frame=_ff_px)
    else:
        # single pass (default, best quality but about 118 GB peak at 480x832x81)
        video = pipe.vae.decode(inputs_shared["latents"], device=dev, tiled=False,
                                first_frame=_ff_px)
    video = pipe.vae_output_to_video(video)
    pipe.load_models_to_device([])
    # return the final latent, for prior-only decoding and other analysis
    if getattr(dual_schedule_generate, "_return_latents", False):
        return video, inputs_shared["latents"].detach().cpu()
    return video


def parse_args():
    p = argparse.ArgumentParser()
    # Checkpoints
    p.add_argument("--dit_checkpoint", type=str, required=True,
                   help="DiT LoRA+patchify+head checkpoint (.safetensors)")
    p.add_argument("--vae_checkpoint", type=str, required=True,
                   help="Geoprior VAE training checkpoint (.ckpt)")
    p.add_argument("--vae_pretrained", type=str,
                   default=os.environ.get("GRACE_WAN_VAE", os.path.join(_WAN_I2V, "Wan2.1_VAE.pth")))
    # VAE config (must match training)
    p.add_argument("--vae_z_dim", type=int, default=32)
    p.add_argument("--vae_prior_z_dim", type=int, default=16)
    # [crossattn] first-frame cross-attention build. It must match the checkpoint's training settings.
    p.add_argument("--ff_window", type=int, default=32, help="crossattn windowed attention window size")
    p.add_argument("--ff_encoder_source", type=str, default="residual", choices=["residual", "base"],
                   help="which encoder supplies the cross-attention K/V: residual (self.encoder) or base (prior_encoder)")
    p.add_argument("--ff_dual_source", action="store_true",
                   help="attach both residual and base encoders for cross-attention; overrides --ff_encoder_source")
    # R2n / ff architecture: turns on stages_after_norm, norm_before_head, expand_encoder_head,
    #   expand_conv2, no_decoder_mirror and b_adaptive together.
    #   * GRACE_CROSSATTN_REPO must point at a training repo that supports R2n, or the builder cannot read it.
    # The c32 (z16) architecture is R2n but turns expand_encoder_head and expand_conv2 **off**.
    #   c128 and c256 have both on (the --vae_r2n default). The c32 launcher passes --no_expand_conv2 and
    #   never passes expand_encoder_head, so leaving them on makes the encoder fail to load on a key mismatch.
    p.add_argument("--vae_no_expand_head", action="store_true",
                   help="c32 variant: within --vae_r2n, turn off expand_encoder_head and expand_conv2")
    p.add_argument("--vae_r2n", action="store_true",
                   help="use the R2n VAE architecture")
    # f32t4 (c64 / c256) architecture. Use together with --vae_r2n.
    #   Four things differ from f16t8: encoder stages (downsample2d x2), decoder stages (upsample2d_keepdim x2,
    #   not before_head), decoder_mirror True, and prior subsample 'bilinear_s4t1_2stage'.
    #   Left unset, the f16t8 path is completely unchanged.
    p.add_argument("--vae_f32t4", action="store_true",
                   help="use the f32t4 architecture; pass together with --vae_r2n")
    # decoder-swap: base (encoder and prior) comes from --vae_checkpoint (EMA) and only the decoder and ff
    #   are overlaid from this checkpoint. Stage 2 trained against the EMA encoder, so loading a derived
    #   checkpoint whole would make the encoder raw and break the y-cond / latent space.
    p.add_argument("--vae_decoder_checkpoint", type=str, default=None,
                   help="checkpoint whose decoder (and first-frame module) is overlaid on the VAE")
    p.add_argument("--zmain_stats_path", type=str,
                   default=os.path.join(_ROOT, "scripts/zmain_stats_v2_caption_freeze_88k.json"))
    # temporal tiled decode - avoids the causal decoder's long-video drift collapse in later frames.
    p.add_argument("--temporal_tile", action="store_true",
                   help="decode long videos in overlapping temporal windows and blend them, which avoids drift")
    p.add_argument("--temporal_tile_size", type=int, default=5, help="temporal tile latent window")
    p.add_argument("--temporal_tile_stride", type=int, default=1, help="temporal tile latent stride (1 = 80%% overlap, smoothest)")
    # [crossattn spatial tile] avoids the ~118 GB single-pass peak with per-tile decode and a CPU blend.
    p.add_argument("--spatial_tile", action="store_true",
                   help="tile the VAE decode spatially to lower peak GPU memory; cross-attention features are cropped per tile")
    p.add_argument("--spatial_tile_size", type=str, default="20,28", help="spatial tile size in latent units, h,w (default 20,28)")
    p.add_argument("--spatial_tile_stride", type=str, default="10,14", help="spatial tile stride in latent units, h,w (default 10,14 = 50%% overlap)")
    # [FrameInit / ConsistI2V] anchor the low frequencies of the initial noise to the first frame (inference-only).
    p.add_argument("--frameinit", action="store_true", help="FrameInit: anchor the low frequencies of the initial noise to the first-frame latent")
    p.add_argument("--frameinit_sigma", type=float, default=0.9, help="initial noise sigma; lower anchors the first frame more strongly, 1.0 is pure noise")
    p.add_argument("--frameinit_d_s", type=float, default=0.25, help="spatial low-frequency cutoff")
    p.add_argument("--frameinit_d_t", type=float, default=0.25, help="temporal low-frequency cutoff")
    p.add_argument("--frameinit_filter", type=str, default="gaussian", choices=["gaussian", "butterworth", "ideal", "box"])
    p.add_argument("--frameinit_target", type=str, default="all", choices=["all", "prior", "main"],
                   help="which channels to anchor: all / prior (base only) / main (residual only)")
    p.add_argument("--frameinit_norm_ff", action="store_true",
                   help="normalize the first-frame latent to unit std per channel")
    p.add_argument("--frameinit_skip_frame0", action="store_true",
                   help="leave frame 0 untouched and anchor only the later frames")
    p.add_argument("--frameinit_truncstart", action="store_true",
                   help="truncated start: begin the ladder at sigma0 = --frameinit_sigma with a matching initial latent")
    p.add_argument("--frameinit_static_encode", action="store_true",
                   help="build the anchor by repeating the first frame and encoding it, instead of repeating its latent")
    # LoRA config (must match training)
    # default matches training (train_dit). With k_img and v_img missing, 160 LoRA keys are silently
    # dropped and the cross-attention image branch runs half-trained - the swirl-artifact incident.
    p.add_argument("--lora_target_modules", type=str, default="q,k,v,o,k_img,v_img,ffn.0,ffn.2")
    p.add_argument("--lora_rank", type=int, default=512)
    p.add_argument("--no_merge_lora", action="store_true",
                    help="keep the LoRA wrappers instead of folding them into the base weights. "
                         "Folding is the default: it is how the latency in the paper was measured "
                         "and what you want for normal use. Folding is NOT bit-exact in bf16, "
                         "though (measured 42.3 dB on one 480x832x81 sample), so pass this flag to "
                         "reproduce the exact videos the paper's VBench scores came from.")
    p.add_argument("--maxq", action="store_true", default=True,
                   help=argparse.SUPPRESS)        # default; kept so old command lines still work
    p.add_argument("--small", dest="maxq", action="store_false",
                   help="write smaller, lossy mp4s instead of the default visually lossless ones")
    p.add_argument("--lossless", action="store_true",
                   help="bit-exact RGB (libx264rgb crf 0, rgb24). FOR METRICS ONLY - gbrp streams "
                        "render with a green cast in browsers and most players. Use --maxq to watch.")
    # Input
    p.add_argument("--image", type=str, default="",
                   help="a single input image; use this instead of --image_dir")
    p.add_argument("--prompt", type=str, default="",
                   help="prompt text; with --image it is the caption, with --image_dir it "
                        "replaces the filename for every image. Defaults to the filename")
    p.add_argument("--image_dir", type=str, default="",
                   help="directory of images; each filename is used as its prompt unless "
                        "--prompt or --prompts_json says otherwise")
    p.add_argument("--prompts_json", type=str, default="",
                   help="VBench format JSON with image_name + prompt_en")
    p.add_argument("--eval_metadata", type=str, default="",
                   help="JSONL with video_path, start_frame_idx, caption fields")
    # Generation config
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--prior_tail_knee", type=float, default=None,
                   help="hold the base ladder at the knee, then descend gently to 0.02 over the remaining steps")
    p.add_argument("--prior_sigma_floor", type=float, default=0.02,
                   help="sigma_min floor for the base ladder (0.02 matches training)")
    p.add_argument("--crop_input", action="store_true", help="center-crop the input image to the output aspect instead of resizing it")
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--output_dir", type=str, required=True)
    # Multi-GPU splitting
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--indices", type=str, default=None,
                   help="comma separated indices into the sorted image_dir; overrides --start_idx / --n_videos")
    p.add_argument("--n_videos", type=int, default=-1, help="-1 = all")
    p.add_argument("--prior_floor_cond", type=float, default=None,
                   help="conditioning value forced on the base branch inside the floor region (default 1000*sigma)")
    p.add_argument("--device", type=str, default="cuda:0")
    # dual_schedule (per-branch timestep shift). Off means the ordinary pipe() path.
    p.add_argument("--dual_schedule", action="store_true",
                   help="denoise the residual and base latents on different noise schedules")
    p.add_argument("--shift_main", type=float, default=5.0, help="beta_main, the flow-matching shift of the main ladder")
    p.add_argument("--shift_prior", type=float, default=5.0, help="single beta_prior, used when no sweep is given")
    p.add_argument("--shift_prior_sweep", type=str, default="",
                   help="comma separated beta_prior list; overrides --shift_prior")
    p.add_argument("--cfg_scale", type=float, default=5.0, help="classifier-free guidance scale")
    p.add_argument("--rope_pos_scale", type=str, default=None,
                   help="'f,h,w' RoPE position scale; must match the value used in training")
    p.add_argument("--init_noise_alpha", type=float, default=0.0,
                   help="strength of time-correlated initial noise (0 = off)")
    p.add_argument("--vscale_main", type=float, default=1.0,
                   help="velocity scaling applied to the residual latent (1.0 = off)")
    p.add_argument("--cfg_main", type=float, default=None,
                   help="separate CFG scale for the residual latent (default: same as --cfg_scale)")
    # [async] base-leading offset sampling, for checkpoints trained with the async band [0.15, 0.55]
    p.add_argument("--async_delta", type=float, default=None,
                   help="asymmetric denoising offset delta, in u space, with the base latent leading")
    p.add_argument("--async_reverse", action="store_true",
                   help="swap the two sigma ladders, for checkpoints trained with the residual ahead")
    p.add_argument("--async_delta_sweep", type=str, default="",
                   help="comma separated delta list; overrides --async_delta")
    # [restart] two-pass: renoise the finished main stream and run again (async mode only)
    p.add_argument("--restart_sigma", type=float, default=None,
                   help="sigma at which a second pass restarts (None = single pass)")
    p.add_argument("--restart_steps", type=int, default=20,
                   help="number of steps in the second pass")
    p.add_argument("--prior_cfg_scale", type=float, default=None,
                   help="separate guidance weight for the base branch")
    p.add_argument("--prior_cfg_null_y", action="store_true",
                   help="also zero the base slice of y in the null pass")
    p.add_argument("--prior_cfg_weak_delta", type=float, default=None,
                   help="renoise the base branch up to this lead for the weak pole (None = zero null)")
    p.add_argument("--prior_cfg_main_only", action="store_true",
                   help="apply the guidance extrapolation to the residual channels only")
    p.add_argument("--save_latents", action="store_true",
                   help="also write the final latent next to the mp4 as .pt")
    p.add_argument("--inject_prior_from", type=str, default=None,
                   help="saved latent (.pt) whose base slice is teacher-forced at every step")
    p.add_argument("--async_shift_prior", type=float, default=None,
                   help="make the two ladders differ by shift instead of offset; this is the base shift")
    p.add_argument("--async_shift_prior_sweep", type=str, default="",
                   help="comma separated shift_prior list; overrides --async_shift_prior")
    p.add_argument("--async_decoupled", type=float, default=None,
                   help="base runs the full ladder while the residual is offset by this amount")
    p.add_argument("--first_frame_inject", action="store_true",
                   help="overwrite frame 0 at every step with the forward-noised clean latent")
    p.add_argument("--first_frame_inject_clean", action="store_true",
                   help="overwrite frame 0 at every step with the clean latent (no noise)")
    # extra negative prompt appended to the default NEG_PROMPT (a face-stabilizing block, for example)
    p.add_argument("--neg_extra", type=str, default=None,
                   help="extra negative prompt appended to NEG_PROMPT")
    return p.parse_args()


def main():
    import time as _time
    _t_start = _time.time()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    global _LOSSLESS, _MAXQ          # [release] hand the save-quality switches to _save()
    _LOSSLESS = bool(getattr(args, "lossless", False))
    _MAXQ = bool(getattr(args, "maxq", False))
    # extend the negative prompt (a face-stabilizing block, for example)
    if args.neg_extra:
        global NEG_PROMPT
        NEG_PROMPT = NEG_PROMPT + ", " + args.neg_extra
        print(f"[neg_extra] NEG_PROMPT += {args.neg_extra[:80]}...")

    # [safety] When sharing a GPU with a training run, this guarantees who dies first: past the cap this
    # process hits OOM and training is protected. With the variable unset it does nothing.
    _mem_frac = os.environ.get("GRACE_GPU_MEM_FRACTION", "")
    if _mem_frac:
        _dev_idx = int(args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_per_process_memory_fraction(float(_mem_frac), device=_dev_idx)
        print(f"[safety] GPU memory fraction cap = {_mem_frac} (device {_dev_idx})")

    # Build item list
    if args.image:
        # [release] One image and one sentence, so this matches the t2v --prompt.
        #   The --image_dir path is untouched, so existing batch runs behave exactly as before.
        if not os.path.isfile(args.image):
            raise SystemExit(f"[FAIL] no such image: {args.image}")
        _stem = os.path.splitext(os.path.basename(args.image))[0]
        all_items = [{"image_path": args.image,
                      "caption": args.prompt or _stem,
                      "name": _stem if len(_stem) <= 100 else _stem[:100].rstrip()}]
    elif args.image_dir:
        # One image can carry **several** prompts.
        #   VBench i2v camera_motion is 109 images x 7 camera instructions = 763 entries.
        #   ("..., camera pans left / pans right / tilts up / tilts down / zooms in / zooms out / static").
        #   prompt_map used to be a plain dict, so the 7 collapsed into the last one, and since the output was
        #   named after the image they overwrote each other anyway. Now it is a list and the filename carries a tag.
        #   An image with a single prompt keeps exactly the old name and behaviour, preserving reproducibility.
        prompt_map = {}
        motion_map = {}
        if args.prompts_json:
            with open(args.prompts_json) as f:
                for ent in json.load(f):
                    if "image_name" in ent and "prompt_en" in ent:
                        prompt_map.setdefault(ent["image_name"], []).append(ent["prompt_en"])
                        # If a camera_motion field is present, build the tag from it.
                        motion_map.setdefault(ent["image_name"], []).append(ent.get("camera_motion"))
        def _slug(pr, stem, motion=None):
            # The tag used to be the first 40 characters of the prompt. The trainstyle camera prompts all begin with
            #   'The video shows the following scene.' and differ only in the **last** sentence, so all 7 collapsed
            #   into one filename: the first (pans left) survived and the other 6 were skipped as already existing.
            #   So camera_motion is used when present, and otherwise the **last** 40 characters of the prompt,
            #   which is the part that differs.
            if motion:
                t = re.sub(r"[^A-Za-z0-9]+", "_", motion).strip("_").lower()
                return t[:40] or None
            t = pr[len(stem):] if pr.startswith(stem) else pr[-60:]
            t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()
            return t[-40:] or None
        img_files = sorted(fn for fn in os.listdir(args.image_dir)
                           if fn.lower().endswith((".jpg", ".jpeg", ".png")))
        all_items = []
        for fn in img_files:
            stem = os.path.splitext(fn)[0]
            # Truncate very long filenames (the ext4/lustre 255-byte limit). The caption used for generation is kept whole.
            safe_name = stem if len(stem) <= 100 else stem[:100].rstrip()
            prs = prompt_map.get(fn) or ([args.prompt] if args.prompt else [stem])
            _mos = motion_map.get(fn) or [None] * len(prs)
            for _k, pr in enumerate(prs):
                _nm = safe_name
                if len(prs) > 1:
                    _sl = _slug(pr, stem, _mos[_k] if _k < len(_mos) else None)
                    _nm = f"{safe_name}__{_sl}" if _sl else f"{safe_name}__p{_k}"
                all_items.append({
                    "image_path": os.path.join(args.image_dir, fn),
                    "caption": pr,
                    "name": _nm,
                })
        _multi = sum(1 for v in prompt_map.values() if len(v) > 1)
        if _multi:
            print(f"[prompt] {_multi} images carry more than one prompt -> {len(all_items)} videos in total "
                  f"(from {len(img_files)} images)")
    elif args.eval_metadata:
        with open(args.eval_metadata) as f:
            all_items = [json.loads(l) for l in f if l.strip()]
    else:
        raise ValueError("Provide --image, --image_dir or --eval_metadata")

    # Slice for multi-GPU
    # --indices: scattered indices in one process, so the model loads once. Only item selection changes.
    if getattr(args, "indices", None):
        _sel = [int(t) for t in args.indices.split(",") if t.strip() != ""]
        items = [all_items[i] for i in _sel]
        print(f"Items: {len(items)} ({len(_sel)} indices selected, total={len(all_items)})")
    elif args.n_videos < 0:
        items = all_items[args.start_idx:]
    else:
        items = all_items[args.start_idx:args.start_idx + args.n_videos]
    if not getattr(args, "indices", None):
        print(f"Items: {len(items)} (start_idx={args.start_idx}, total={len(all_items)})")

    # Load pipeline
    print("\nLoading pipeline...")
    model_paths = build_model_paths()
    tokenizer_path = os.path.join(_WAN_I2V, "google/umt5-xxl")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=args.device,
        model_configs=[ModelConfig(path) for path in model_paths],
        tokenizer_config=ModelConfig(tokenizer_path),
    )

    # Replace the VAE with the first_frame_inject-compatible loader (only the decoder is swapped).
    #   base (encoder and prior) stays the same; only the decoder becomes crossattn (ff_inject), which
    print("\nLoading Geoprior VAE (crossattn / first_frame_inject)...")
    pipe.vae = load_grace_geoprior_vae_crossattn(
        checkpoint_path=args.vae_checkpoint,
        pretrained_path=args.vae_pretrained,
        z_dim=args.vae_z_dim, prior_z_dim=args.vae_prior_z_dim,
        # [f32t4 / c64] the architecture differs from f16t8. These map one to one onto the training launcher flags:
        #   f16t8(c32/c128/c256 3d): downsample3d x1 / upsample3d(before_head) / decoder_mirror False
        #   f32t4(c64):              downsample2d x2 / upsample2d_keepdim x2(add_decoder_stages) / mirror True
        add_encoder_stages=([{"mode": "downsample2d", "num_res_blocks": 2, "init": "zero"}] * 2
                            if args.vae_f32t4 else
                            [{"mode": "downsample3d", "num_res_blocks": 2, "init": "zero"}]),
        add_decoder_before_head_stages=(None if args.vae_f32t4 else
                                        [{"mode": "upsample3d", "num_res_blocks": 2}]),
        **({"add_decoder_stages": [{"mode": "upsample2d_keepdim", "num_res_blocks": 2, "init": "zero"}] * 2}
           if args.vae_f32t4 else {}),
        # [R2n] this lineage has expand_conv2=True (verified with missing=0 in a reconstruction eval),
        #   so no_expand_conv2=False under R2n. Left unset it stays True as before.
        no_expand_conv2=(True if args.vae_no_expand_head else (not args.vae_r2n)),
        # crossattn build matching the training settings: window 32, residual encoder, injection at every level
        first_frame_inject=True,
        ff_window=args.ff_window,
        ff_encoder_source=args.ff_encoder_source,
        ff_dual_source=args.ff_dual_source,   # [dual] residual and base encoders at once
        ff_inject_levels='all',
        device=args.device, dtype=torch.bfloat16,
        zmain_stats_path=args.zmain_stats_path, zprior_norm=True,
        # [R2n] architecture set, exactly as the launch shell passes it:
        #   stages_after_norm + stages_norm_before_head + expand_encoder_head + --no_decoder_mirror).
        #   use_b_adaptive is excluded - the numbers equal plain conv (same weights; the dual view is a backward
        #   trick) and the dual-view tuple breaks the wrapper in train-mode encode.
        # [f32t4 / c64] decoder_mirror is True because the c64 launcher does not pass --vae_no_decoder_mirror
        #   (the f16t8 lineage does, making it False). That one flag decides the decoder architecture.
        **({"stages_after_norm": True, "stages_norm_before_head": True,
            "expand_encoder_head": (not args.vae_no_expand_head),
            "decoder_mirror": bool(args.vae_f32t4)} if args.vae_r2n else {}),
        **({"decoder_checkpoint_path": args.vae_decoder_checkpoint} if args.vae_decoder_checkpoint else {}),
    )
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    # Forcing a single-pass encode (vae.model.train()) was tried and withdrawn.
    #   train_dit.py calls pipe.vae.eval() at setup and keeps it there, so stage 2's y-cond and latent targets
    #   are all eval-chunked encodes; inference matching stage 2 means eval (chunked) as well. The latent
    #   difference between chunked and single is 0.5%, and single-pass is impossible on an 80 GB GPU anyway
    #   because the first conv falls back to im2col at 81 frames and 480x832 (a single 156 GiB allocation).
    # temporal tiled decode flag (fixes long-video drift collapse)
    pipe.vae.temporal_tile = args.temporal_tile
    pipe.vae.temporal_tile_size = args.temporal_tile_size
    pipe.vae.temporal_tile_stride = args.temporal_tile_stride
    if args.temporal_tile:
        print(f"[temporal_tile] ON — latent win={args.temporal_tile_size} stride={args.temporal_tile_stride}")
    # [crossattn spatial tile] a single-pass decode at 480x832x81 peaks near 118 GB and runs out of memory.
    #   Spatial tiling splits the latent into tiles, decodes each (cropping the first frame per tile) and
    #   blends on the CPU, which lowers the GPU peak to one tile. crossattn overrides _decode_tile for the crop.
    pipe.vae._use_spatial_tile = args.spatial_tile
    pipe.vae._spatial_tile_size = tuple(int(x) for x in args.spatial_tile_size.split(","))
    pipe.vae._spatial_tile_stride = tuple(int(x) for x in args.spatial_tile_stride.split(","))
    if args.spatial_tile:
        print(f"[spatial_tile] ON — latent tile={pipe.vae._spatial_tile_size} stride={pipe.vae._spatial_tile_stride} (lowers decode peak memory)")
    # [FrameInit] passed as pipe attributes (dual_schedule_generate reads them with getattr)
    pipe._prior_tail_knee = args.prior_tail_knee
    pipe._prior_sigma_floor = args.prior_sigma_floor  # passed through as the async_ladders sigma_min
    pipe._use_frameinit = args.frameinit
    pipe._frameinit_sigma = args.frameinit_sigma
    pipe._frameinit_d_s = args.frameinit_d_s
    pipe._frameinit_d_t = args.frameinit_d_t
    pipe._frameinit_filter = args.frameinit_filter
    pipe._frameinit_target = args.frameinit_target
    pipe._frameinit_norm_ff = args.frameinit_norm_ff
    pipe._frameinit_skip_frame0 = args.frameinit_skip_frame0
    pipe._frameinit_truncstart = args.frameinit_truncstart
    pipe._frameinit_static_encode = args.frameinit_static_encode
    if args.frameinit:
        print(f"[frameinit] ON target={args.frameinit_target} sigma={args.frameinit_sigma} d_s={args.frameinit_d_s} d_t={args.frameinit_d_t} filter={args.frameinit_filter}")
    # Match training: override subsample_mode
    #   f32t4 (c64) uses 'bilinear_s4t1_2stage', the same value as the training launcher.
    #   The f16t8 lineage stays on 'bilinear'. Getting this wrong builds z_prior from a different distribution.
    _SUB = 'bilinear_s4t1_2stage' if args.vae_f32t4 else 'bilinear'
    if hasattr(pipe.vae.model, 'subsample_mode'):
        pipe.vae.model.subsample_mode = _SUB
        print(f"[vae] subsample_mode = {_SUB}")
    pipe.height_division_factor = pipe.vae.upsampling_factor * 2
    pipe.width_division_factor = pipe.vae.upsampling_factor * 2
    pipe.time_division_factor = pipe.vae.temporal_factor
    pipe.time_division_remainder = 1
    print(f"VAE: spatial={pipe.vae.upsampling_factor}x, temporal={pipe.vae.temporal_factor}x, "
          f"latent_channels={pipe.vae.latent_channels}")

    # Setup DA-VAE DiT (patchify + head + LoRA + load checkpoint)
    print("\nSetting up DA-VAE DiT...")
    # [async] fix the delta list (a sweep wins). None turns async off entirely, and t2 is not even built.
    if args.async_delta_sweep.strip():
        _async_deltas = [float(x) for x in args.async_delta_sweep.split(",")]
    elif args.async_delta is not None:
        _async_deltas = [args.async_delta]
    else:
        _async_deltas = None
    # shift-based async instead of an offset - when given, t2_projection must also be built
    if args.async_shift_prior_sweep.strip():
        _async_shift_priors = [float(x) for x in args.async_shift_prior_sweep.split(",")]
    elif args.async_shift_prior is not None:
        _async_shift_priors = [args.async_shift_prior]
    else:
        _async_shift_priors = None
    _async_decoupled = args.async_decoupled   # [NEW] decoupled (prior full [1,0], main [1+offset,0])

    setup_davae_dit(
        pipe, args.dit_checkpoint,
        vae_z_dim=args.vae_z_dim, vae_prior_z_dim=args.vae_prior_z_dim,
        lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank,
        use_dual_schedule=args.dual_schedule,
        use_async=(_async_deltas is not None) or (_async_shift_priors is not None) or (_async_decoupled is not None),
    )
    if not args.no_merge_lora:
        fold_lora(pipe.dit)   # default. It is what the paper's latency assumes, and bf16 rounding makes it not bit-exact

    # rope and init-noise options are wired in the args scope (main).
    #   They used to sit inside setup_davae_dit, where a NameError made the alpha sweep and the rope probe fail silently.
    if getattr(args, "rope_pos_scale", None):
        _rps = tuple(float(v) for v in args.rope_pos_scale.split(","))
        _rps = tuple(int(v) if float(v).is_integer() else v for v in _rps)
        pipe.dit.rope_pos_scale = _rps
        print(f"[rope-scale] dit.rope_pos_scale={_rps}", flush=True)
    pipe._init_noise_alpha = float(getattr(args, "init_noise_alpha", 0.0) or 0.0)
    if pipe._init_noise_alpha > 0:
        print(f"[init-noise] temporal-correlated alpha={pipe._init_noise_alpha}", flush=True)

    # Generation
    _t_load = _time.time() - _t_start
    print(f"[TIMING] Total load (pipeline+VAE+DiT setup): {_t_load:.1f}s", flush=True)
    print(f"\nGenerating {len(items)} videos...")
    _t_gen_start = _time.time()

    # [async] async branch: delta values x items, saved into subdirectories (same pattern as dual_schedule).
    # Both branches share shift_main (default 5.0); asynchrony comes from the offset alone.
    # decoupled branch: prior runs the full [1,0] ladder, main runs [1+offset,0].
    if _async_decoupled is not None:
        print(f"[async-decoupled] offset={_async_decoupled}, shift_main={args.shift_main}, cfg={args.cfg_scale}")
        sub = os.path.join(args.output_dir, f"decoupled{_async_decoupled:.2f}_main{args.shift_main:.1f}")
        os.makedirs(sub, exist_ok=True)
        for i, item in enumerate(items):
            if "image_path" in item:
                image_path = item["image_path"]; prompt = item.get("caption", "")
                name = item.get("name") or os.path.splitext(os.path.basename(image_path))[0]
            else:
                video_path = item["video_path"]; start_frame_idx = int(item.get("start_frame_idx", 0))
                prompt = item.get("caption_gpt_4o", "") or item.get("caption", "")
                name = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(sub, f"{name}.mp4")
            if os.path.exists(out_path):
                print(f"[decoupled{_async_decoupled:.2f}] [{i+1}/{len(items)}] {name} [skip]"); continue
            try:
                if "image_path" in item:
                    input_image = Image.open(image_path).convert("RGB")
                else:
                    reader = imageio.get_reader(video_path); raw = reader.get_data(start_frame_idx); reader.close()
                    input_image = Image.fromarray(raw)
                # --crop_input: aspect-preserving center crop, then the downstream resize.
                #   The default resize distorts the aspect (3:2 becomes 1.733). This is opt-in, so old runs reproduce.
                # [guard] Stops a silent aspect mismatch.
                #   What happened: a 3:2 (1.5000) image set was run at 832x480 (1.7333) without --crop_input and the
                #   first frame came out stretched 15.6% horizontally. Training preprocessing (ImageCropAndResize) is an
                #   aspect-preserving resize plus center crop, so **stretched geometry is outside the training
                #   distribution**. Nothing crashed, so about 600 videos were generated before anyone noticed.
                #   The 3% tolerance passes 7:4 (0.96%) and 16:9 (2.56%) and blocks 3:2 (15.6%) and 8:5 (7.7%).
                _w0, _h0 = input_image.size; _tr0 = args.width / args.height
                _dev0 = abs(_w0 / _h0 - _tr0) / _tr0
                if _dev0 > 0.03 and not getattr(args, "crop_input", False):
                    raise SystemExit(
                        f"[FAIL] input image aspect {_w0/_h0:.4f} differs from output aspect {_tr0:.4f} "
                        f"by {_dev0*100:.1f}% (3% allowed). Pass --crop_input, or "
                        f"use an --image_dir whose images match the output aspect. Current dir={args.image_dir}")
                if getattr(args, "crop_input", False):
                    _w, _h = input_image.size; _tr = args.width / args.height
                    if _w / _h > _tr:
                        _nw = int(_h * _tr); input_image = input_image.crop(((_w-_nw)//2, 0, (_w+_nw)//2, _h))
                    else:
                        _nh = int(_w / _tr); input_image = input_image.crop((0, (_h-_nh)//2, _w, (_h+_nh)//2))
            except Exception as e:
                print(f"  [skip] image load error: {e}"); continue
            with torch.no_grad():
                frames = dual_schedule_generate(
                    pipe, prompt, NEG_PROMPT, input_image,
                    args.height, args.width, args.num_frames, args.num_inference_steps,
                    args.seed, args.shift_main, args.shift_main, args.vae_z_dim,
                    tiled=True, cfg_scale=args.cfg_scale,
                    vscale_main=args.vscale_main, async_decoupled=_async_decoupled,
                    first_frame_inject=args.first_frame_inject)
            _save(frames, out_path, fps=16)
            print(f"[decoupled{_async_decoupled:.2f}] [{i+1}/{len(items)}] saved {name}", flush=True)
        print(f"\nDone (decoupled)! {args.output_dir}")
        return

    # shift-based async branch (instead of an offset, so no steps are wasted). shift_main stays 5 and only the base moves.
    if _async_shift_priors is not None:
        print(f"[async-shift] shift_main={args.shift_main}, shift_prior={_async_shift_priors}, cfg={args.cfg_scale}")
        for sp in _async_shift_priors:
            sub = os.path.join(args.output_dir, f"shiftP{sp:.2f}_main{args.shift_main:.1f}")
            os.makedirs(sub, exist_ok=True)
            for i, item in enumerate(items):
                if "image_path" in item:
                    image_path = item["image_path"]; prompt = item.get("caption", "")
                    name = item.get("name") or os.path.splitext(os.path.basename(image_path))[0]
                else:
                    video_path = item["video_path"]; start_frame_idx = int(item.get("start_frame_idx", 0))
                    prompt = item.get("caption_gpt_4o", "") or item.get("caption", "")
                    name = os.path.splitext(os.path.basename(video_path))[0]
                out_path = os.path.join(sub, f"{name}.mp4")
                if os.path.exists(out_path):
                    print(f"[shiftP{sp:.2f}] [{i+1}/{len(items)}] {name} [skip]"); continue
                try:
                    if "image_path" in item:
                        input_image = Image.open(image_path).convert("RGB")
                    else:
                        reader = imageio.get_reader(video_path); raw = reader.get_data(start_frame_idx); reader.close()
                        input_image = Image.fromarray(raw)
                except Exception as e:
                    print(f"  [skip] image load error: {e}"); continue
                with torch.no_grad():
                    frames = dual_schedule_generate(
                        pipe, prompt, NEG_PROMPT, input_image,
                        args.height, args.width, args.num_frames, args.num_inference_steps,
                        args.seed, args.shift_main, args.shift_main, args.vae_z_dim,
                        tiled=True, cfg_scale=args.cfg_scale,
                        vscale_main=args.vscale_main, async_shift_prior=sp,
                        first_frame_inject=args.first_frame_inject)
                _save(frames, out_path, fps=16)
                print(f"[shiftP{sp:.2f}] [{i+1}/{len(items)}] saved {name}", flush=True)
        print(f"[TIMING] Total wall: {_time.time() - _t_start:.1f}s", flush=True)
        print(f"\nDone (async-shift)! {args.output_dir}")
        return

    if _async_deltas is not None:
        print(f"[async] delta={_async_deltas}, shift={args.shift_main}, cfg={args.cfg_scale}" + (" reverse=True (main is the clean one: ladders swapped)" if args.async_reverse else ""))
        # the donor latent is loaded once and only the base slice is passed on
        _inj_zp = None
        if args.inject_prior_from:
            _inj_zp = torch.load(args.inject_prior_from, map_location="cpu")[:, args.vae_z_dim:]
            print(f"[inject] wrong-prior injection: {os.path.basename(args.inject_prior_from)} -> {tuple(_inj_zp.shape)}")
        for ad in _async_deltas:
            sub = os.path.join(args.output_dir, f"asyncD{ad:.2f}_shift{args.shift_main:.1f}" + ("_REV" if args.async_reverse else ""))
            os.makedirs(sub, exist_ok=True)
            for i, item in enumerate(items):
                if "image_path" in item:
                    image_path = item["image_path"]
                    prompt = item.get("caption", "")
                    name = item.get("name") or os.path.splitext(os.path.basename(image_path))[0]
                else:
                    video_path = item["video_path"]
                    start_frame_idx = int(item.get("start_frame_idx", 0))
                    prompt = item.get("caption_gpt_4o", "") or item.get("caption", "")
                    name = os.path.splitext(os.path.basename(video_path))[0]
                out_path = os.path.join(sub, f"{name}.mp4")
                if os.path.exists(out_path):
                    print(f"[asyncD{ad:.2f}] [{i+1}/{len(items)}] {name} [skip]"); continue
                try:
                    if "image_path" in item:
                        input_image = Image.open(image_path).convert("RGB")
                    else:
                        reader = imageio.get_reader(video_path)
                        raw = reader.get_data(start_frame_idx); reader.close()
                        input_image = Image.fromarray(raw)
                except Exception as e:
                    print(f"  [skip] image load error: {e}"); continue
                with torch.no_grad():
                    dual_schedule_generate._return_latents = args.save_latents
                    frames = dual_schedule_generate(
                        pipe, prompt, NEG_PROMPT, input_image,
                        args.height, args.width, args.num_frames, args.num_inference_steps,
                        args.seed, args.shift_main, args.shift_main, args.vae_z_dim,
                        tiled=True, cfg_scale=args.cfg_scale,
                        vscale_main=args.vscale_main, async_delta=ad, async_reverse=args.async_reverse,
                        first_frame_inject=args.first_frame_inject,
                        first_frame_inject_clean=args.first_frame_inject_clean,
                        cfg_main=args.cfg_main,
                        restart_sigma=args.restart_sigma, restart_steps=args.restart_steps,
                        prior_cfg_scale=args.prior_cfg_scale, prior_cfg_null_y=args.prior_cfg_null_y,
                        prior_cfg_weak_delta=args.prior_cfg_weak_delta,
                        prior_cfg_main_only=args.prior_cfg_main_only,
                        prior_floor_cond=args.prior_floor_cond,
                        inject_prior_latents=_inj_zp)
                # with --save_latents, a (video, latents) tuple comes back
                if args.save_latents:
                    frames, _lat = frames
                    torch.save(_lat, out_path.replace(".mp4", "_latents.pt"))
                # Same as the V-RAE gFVD protocol (runtime.py:_atomic_mp4 - imageio quality=9, no ffmpeg_params).
                #   Passing -crf 12 would override quality and break the protocol.
                _save(frames, out_path, fps=16)
                print(f"[asyncD{ad:.2f}] [{i+1}/{len(items)}] saved {name}", flush=True)
        print(f"[TIMING] Total wall: {_time.time() - _t_start:.1f}s", flush=True)
        print(f"\nDone (async)! {args.output_dir}")
        return

    # dual_schedule branch: beta_prior values (a sweep or a single one) x items, saved into subdirectories.
    # With --dual_schedule unset this whole block is skipped and the ordinary pipe() loop below runs.
    if args.dual_schedule:
        if args.shift_prior_sweep.strip():
            prior_betas = [float(x) for x in args.shift_prior_sweep.split(",")]
        else:
            prior_betas = [args.shift_prior]
        print(f"[dual_schedule] shift_main={args.shift_main}, shift_prior={prior_betas}, cfg={args.cfg_scale}")
        for sp in prior_betas:
            sub = os.path.join(args.output_dir, f"beta_main{args.shift_main:.1f}_prior{sp:.1f}")
            os.makedirs(sub, exist_ok=True)
            for i, item in enumerate(items):
                if "image_path" in item:
                    image_path = item["image_path"]
                    prompt = item.get("caption", "")
                    name = item.get("name") or os.path.splitext(os.path.basename(image_path))[0]
                else:
                    video_path = item["video_path"]
                    start_frame_idx = int(item.get("start_frame_idx", 0))
                    prompt = item.get("caption_gpt_4o", "") or item.get("caption", "")
                    name = os.path.splitext(os.path.basename(video_path))[0]
                out_path = os.path.join(sub, f"{name}.mp4")
                if os.path.exists(out_path):
                    print(f"[main{args.shift_main} prior{sp}] [{i+1}/{len(items)}] {name} [skip]"); continue
                try:
                    if "image_path" in item:
                        input_image = Image.open(image_path).convert("RGB")
                    else:
                        reader = imageio.get_reader(video_path)
                        raw = reader.get_data(start_frame_idx); reader.close()
                        input_image = Image.fromarray(raw)
                except Exception as e:
                    print(f"  [skip] image load error: {e}"); continue
                with torch.no_grad():
                    frames = dual_schedule_generate(
                        pipe, prompt, NEG_PROMPT, input_image,
                        args.height, args.width, args.num_frames, args.num_inference_steps,
                        args.seed, args.shift_main, sp, args.vae_z_dim, tiled=True, cfg_scale=args.cfg_scale,
                        vscale_main=args.vscale_main)
                _save(frames, out_path, fps=16)
                print(f"[main{args.shift_main} prior{sp}] [{i+1}/{len(items)}] saved {name}", flush=True)
        print(f"[TIMING] Total wall: {_time.time() - _t_start:.1f}s", flush=True)
        print(f"\nDone (dual_schedule)! {args.output_dir}")
        return

    for i, item in enumerate(items):
        if "image_path" in item:
            image_path = item["image_path"]
            prompt = item.get("caption", "")
            name = item.get("name") or os.path.splitext(os.path.basename(image_path))[0]
        else:
            video_path = item["video_path"]
            start_frame_idx = int(item.get("start_frame_idx", 0))
            prompt = item.get("caption_gpt_4o", "") or item.get("caption", "")
            name = os.path.splitext(os.path.basename(video_path))[0]

        out_path = os.path.join(args.output_dir, f"{name}.mp4")
        if os.path.exists(out_path):
            print(f"[{i+1}/{len(items)}] {name} [skip: exists]")
            continue

        print(f"[{i+1}/{len(items)}] {name}")
        print(f"  prompt: {prompt[:120]}...")

        try:
            if "image_path" in item:
                input_image = Image.open(image_path).convert("RGB")
            else:
                reader = imageio.get_reader(video_path)
                raw = reader.get_data(start_frame_idx)
                reader.close()
                input_image = Image.fromarray(raw)
        except Exception as e:
            print(f"  [skip] image load error: {e}")
            continue

        _t_infer_start = _time.time()
        with torch.no_grad():
            # The pure path uses dual_schedule_generate instead of pipe(). Why:
            #   Symptom: running pure (no delta) ran out of memory in the 480x832x81 decode even with SPATIAL_TILE=1,
            #         asking for another 23.14 GiB, and produced no videos at all.
            #   Cause: pipe() calls vae.decode(latents, tiled=True, tile_size=(30,52), tile_stride=...) in
            #         DiffSynth's wan_video.py:353, but our wrapper (grace_video_vae_crossattn.decode) does not
            #         consume those three kwargs - it only reads its own spatial_tiled - so it always took the
            #         single-pass path. Tiled decode, the DiT CPU offload and the first-frame injection all live
            #         inside dual_schedule_generate, which is why async runs were fine and only pure leaked here.
            #   Fix: keep the verified path and switch the dual timestep off instead.
            #         - shift_prior = shift_main makes the two ladders identical, so dual_step reduces to a plain Euler step
            #           (see :365, 'with shift_main == shift_prior this equals the single-schedule pipe()').
            #         - async_delta, async_shift_prior and async_decoupled are not passed, so _pass_t2=False and
            #           neither t2_projection nor head.t_prior_emb is used - a pure single timestep.
            frames = dual_schedule_generate(
                pipe, prompt, NEG_PROMPT, input_image,
                args.height, args.width, args.num_frames, args.num_inference_steps,
                args.seed, args.shift_main, args.shift_main, args.vae_z_dim,
                tiled=True, cfg_scale=args.cfg_scale, vscale_main=args.vscale_main)

        _t_infer = _time.time() - _t_infer_start
        print(f"  [TIMING] pipe() inference (excluding save): {_t_infer:.1f}s", flush=True)
        _t_save_start = _time.time()
        _save(frames, out_path, fps=16)
        _t_save = _time.time() - _t_save_start
        print(f"  [TIMING] save_video: {_t_save:.1f}s", flush=True)
        print(f"  saved: {out_path}")
    _t_gen = _time.time() - _t_gen_start
    print(f"\n[TIMING] Generation ({len(items)} video, incl. save): {_t_gen:.1f}s ({_t_gen/max(len(items),1):.1f}s/video)", flush=True)
    print(f"[TIMING] Total wall: {_time.time() - _t_start:.1f}s (load={_t_load:.1f}s, gen={_t_gen:.1f}s)", flush=True)
    print(f"\nDone! {args.output_dir}")


if __name__ == "__main__":
    main()
