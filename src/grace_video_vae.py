# [Source: DiffSynth-Studio] diffsynth/models/wan_video_vae.py - the WanVideoVAE class
# [modified] integrates GRACE's WanVAE_ and adds checkpoint loading.
# Every change is marked [modified] or [new].
import logging
import os
import sys

import torch
import torch.nn as nn
from einops import repeat, rearrange
from tqdm import tqdm

# ---------------------------------------------------------------------------
# [new] import the GRACE modules
# ---------------------------------------------------------------------------
_GRACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _GRACE_ROOT not in sys.path:
    sys.path.insert(0, _GRACE_ROOT)

from grace import WanVAE_  # noqa: E402
from grace_geoprior import _video_vae_geoprior  # noqa: E402


# ---------------------------------------------------------------------------
# [new] compression factor computation
# ---------------------------------------------------------------------------
@staticmethod
def _compute_temporal_factor(temperal_downsample, add_encoder_stages):
    """Temporal compression factor (raw Wan=4, +1 downsample3d=8)"""
    tf = 1
    for td in temperal_downsample:
        if td:
            tf *= 2
    if add_encoder_stages:
        for stage_cfg in add_encoder_stages:
            if stage_cfg["mode"] in ("downsample3d", "downsample_temporal"):
                tf *= 2
    return tf


@staticmethod
def _compute_spatial_factor(dim_mult, add_encoder_stages):
    """Spatial compression factor (raw Wan=8, +1 downsample3d=16)"""
    sf = 2 ** (len(dim_mult) - 1)
    if add_encoder_stages:
        for stage_cfg in add_encoder_stages:
            if stage_cfg["mode"] in ("downsample3d", "downsample2d"):
                sf *= 2
    return sf


# ---------------------------------------------------------------------------
# [modified] GRACEVideoVAE
# Original: WanVideoVAE from DiffSynth-Studio (wan_video_vae.py:1058-1268)
# Changes:
#   - class name: WanVideoVAE -> GRACEVideoVAE
#   - inner model: VideoVAE_ -> GRACE's WanVAE_ (supports add_stages)
#   - upsampling_factor: hardcoded 8 -> computed
#   - temporal_factor: new attribute
#   - single_encode: handles the (mu, log_var) return
#   - tiled_encode/tiled_decode: hardcoded temporal factor (4) -> computed
# ---------------------------------------------------------------------------
class GRACEVideoVAE(nn.Module):

    # [modified] original: def __init__(self, z_dim=16):
    # Change: added the add_encoder_stages / add_decoder_stages arguments
    def __init__(self, z_dim=16, add_encoder_stages=None, add_decoder_stages=None):
        super().__init__()

        # [new] default: add one downsample3d/upsample3d stage
        if add_encoder_stages is None:
            add_encoder_stages = [{"mode": "downsample3d", "num_res_blocks": 2, "init": "pretrained_copy"}]
        if add_decoder_stages is None:
            add_decoder_stages = [{"mode": "upsample3d", "num_res_blocks": 2, "init": "pretrained_copy"}]
        self.add_encoder_stages = add_encoder_stages
        self.add_decoder_stages = add_decoder_stages

        # --- same as the original: latent normalisation ---
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        # [new] registering as a buffer is what makes .to(device) move it
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('inv_std', torch.tensor([1.0 / s for s in std]))
        # scale is looked up at encode/decode time via _get_scale(device)
        self.scale = None  # deprecated; use _get_scale(device) instead

        # [modified] original: self.model = VideoVAE_(z_dim=z_dim).eval().requires_grad_(False)
        # Change: use GRACE's WanVAE_ (supports add_stages)
        _temperal_downsample = [False, True, True]
        _dim_mult = [1, 2, 4, 4]
        self.model = WanVAE_(
            dim=96,
            z_dim=z_dim,
            dim_mult=_dim_mult,
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=_temperal_downsample,
            dropout=0.0,
            add_encoder_stages=add_encoder_stages,
            add_decoder_stages=add_decoder_stages,
        ).eval().requires_grad_(False)

        # [modified] original: self.upsampling_factor = 8 (hardcoded)
        # Change: computed from add_stages
        self.temporal_factor = _compute_temporal_factor(_temperal_downsample, add_encoder_stages)
        self.upsampling_factor = _compute_spatial_factor(_dim_mult, add_encoder_stages)
        self.z_dim = z_dim
        self.latent_channels = z_dim  # used for the DiT noise shape

    def _get_scale(self, device):
        """[new] Return the scale [mean, inv_std] on the right device."""
        return [self.mean.to(device), self.inv_std.to(device)]

    # --- same as the original: build_1d_mask (wan_video_vae.py:1081-1087) ---
    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    # --- same as the original: build_mask (wan_video_vae.py:1090-1100) ---
    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])
        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)
        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    # --- based on the original, 2 lines changed: tiled_decode (wan_video_vae.py:1103-1152) ---
    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        # [modified] original: out_T = T * 4 - 3
        out_T = T * self.temporal_factor - (self.temporal_factor - 1)
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor), dtype=hidden_states.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self._get_scale(computation_device)).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) * self.upsampling_factor, (size_w - stride_w) * self.upsampling_factor)
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :, :, :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :, :, :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values

    # --- based on the original, 2 lines changed: tiled_encode (wan_video_vae.py:1155-1203) ---
    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if (h-stride_h >= 0 and h-stride_h+size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w-stride_w >= 0 and w-stride_w+size_w >= W): continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        # [modified] original: out_T = (T + 3) // 4
        out_T = (T + self.temporal_factor - 1) // self.temporal_factor
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)
        values = torch.zeros((1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor), dtype=video.dtype, device=data_device)

        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            # [modified] original: hidden_states_batch = self.model.encode(...).to(data_device)
            # Change: GRACE's WanVAE_.encode() returns (mu, log_var)
            hidden_states_batch, _ = self.model.encode(hidden_states_batch, self._get_scale(computation_device))
            hidden_states_batch = hidden_states_batch.to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) // self.upsampling_factor, (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :, :, :,
                target_h:target_h + hidden_states_batch.shape[3],
                target_w:target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :, :, :,
                target_h: target_h + hidden_states_batch.shape[3],
                target_w: target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values

    @property
    def _dtype(self):
        return next(self.model.parameters()).dtype

    # --- based on the original, 1 line changed: single_encode (wan_video_vae.py:1206-1209) ---
    def single_encode(self, video, device):
        video = video.to(device=device, dtype=self._dtype)
        # [modified] original: x = self.model.encode(video, self.scale)
        # Change: GRACE's WanVAE_.encode() returns (mu, log_var)
        mu, _log_var = self.model.encode(video, self._get_scale(device))
        return mu

    # --- same as the original: single_decode (wan_video_vae.py:1212-1215) ---
    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device=device, dtype=self._dtype)
        video = self.model.decode(hidden_state, self._get_scale(device))
        return video.clamp_(-1, 1).float()

    # --- same as the original: encode (wan_video_vae.py:1218-1232) ---
    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
                tile_stride = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    # --- same as the original: decode (wan_video_vae.py:1235-1247) ---
    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            video = video.squeeze(0)
            videos.append(video)
        videos = torch.stack(videos)
        return videos

    # --- same as the original: encode_framewise (wan_video_vae.py:1250-1255) ---
    def encode_framewise(self, videos, device):
        hidden_states = []
        for i in range(videos.shape[2]):
            hidden_states.append(self.single_encode(videos[:, :, i:i+1], device))
        hidden_states = torch.concat(hidden_states, dim=2)
        return hidden_states

    # --- same as the original: decode_framewise (wan_video_vae.py:1258-1263) ---
    def decode_framewise(self, hidden_states, device):
        video = []
        for i in range(hidden_states.shape[2]):
            video.append(self.single_decode(hidden_states[:, :, i:i+1], device))
        video = torch.concat(video, dim=2)
        return video

    # [modified] original: return WanVideoVAEStateDictConverter()
    @staticmethod
    def state_dict_converter():
        return GRACEVideoVAEStateDictConverter()


# ---------------------------------------------------------------------------
# [modified] State dict converter
# Original: WanVideoVAEStateDictConverter (wan_video_vae.py:1271-1282)
# Change: renamed the class and added a from_grace() method
# ---------------------------------------------------------------------------
class GRACEVideoVAEStateDictConverter:

    def __init__(self):
        pass

    # --- same as the original: from_civitai ---
    def from_civitai(self, state_dict):
        state_dict_ = {}
        if 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        for name in state_dict:
            state_dict_['model.' + name] = state_dict[name]
        return state_dict_

    # [new] load from a GRACE training checkpoint
    # Checkpoint layout: {"state_dict": {"gen_model": {...}}, "ema_state_dict": {...}}
    def from_grace(self, state_dict):
        if "state_dict" in state_dict and "gen_model" in state_dict["state_dict"]:
            raw = state_dict["state_dict"]["gen_model"]
        elif "gen_model" in state_dict:
            raw = state_dict["gen_model"]
        else:
            raw = state_dict
        state_dict_ = {}
        for name in raw:
            state_dict_["model." + name] = raw[name]
        return state_dict_


# ---------------------------------------------------------------------------
# [new] GRACE VAE checkpoint loader
# ---------------------------------------------------------------------------
def load_grace_vae(
    checkpoint_path=None,
    pretrained_path=None,
    z_dim=16,
    add_encoder_stages=None,
    add_decoder_stages=None,
    device="cpu",
    dtype=torch.float32,
):
    """Load a GRACEVideoVAE from a GRACE training checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to a GRACE .ckpt file (e.g. checkpoint-17000.ckpt).
    pretrained_path : str, optional
        Path to original Wan2.1_VAE.pth to load base weights first.
    """
    logging.info(f"[GRACE] Building GRACEVideoVAE (z_dim={z_dim})")
    vae = GRACEVideoVAE(
        z_dim=z_dim,
        add_encoder_stages=add_encoder_stages,
        add_decoder_stages=add_decoder_stages,
    )
    converter = GRACEVideoVAEStateDictConverter()

    # 1) load the pretrained base weights (add_stage keys are allowed to be missing)
    if pretrained_path is not None:
        logging.info(f"[GRACE] Loading pretrained base weights from {pretrained_path}")
        base_sd = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        base_sd = converter.from_civitai(base_sd)
        missing, unexpected = vae.load_state_dict(base_sd, strict=False)
        logging.info(f"[GRACE]   Base: {len(missing)} missing (expected for add_stages), "
                     f"{len(unexpected)} unexpected")

    # 2) load the GRACE checkpoint (add_stage included) - None means base VAE only
    if checkpoint_path is not None:
        logging.info(f"[GRACE] Loading GRACE checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_sd = converter.from_grace(ckpt)

        # 3) merge the EMA weights when present (EMA is more stable)
        ema_sd = ckpt.get("ema_state_dict", {})
        if ema_sd:
            logging.info(f"[GRACE]   Found EMA weights — merging into checkpoint")
            for k, v in ema_sd.items():
                ckpt_sd["model." + k] = v

        missing, unexpected = vae.load_state_dict(ckpt_sd, strict=False)
        logging.info(f"[GRACE]   GRACE ckpt: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            logging.warning(f"[GRACE]   Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        logging.info(f"[GRACE] checkpoint_path=None — base pretrained VAE only")

    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae


# ---------------------------------------------------------------------------
# [new] GRACEGeopriorVideoVAE
# Wraps a geoprior model (dual_branch + prior_encoder) in the WanVideoVAE interface.
# encode() returns cat([z_main, z_prior]); decode() takes that concatenated z.
# scale=None (eval only, no DiT normalization needed).
# ---------------------------------------------------------------------------
class GRACEGeopriorVideoVAE(nn.Module):

    def __init__(self, model, zmain_stats_path=None, zprior_norm=True, add_encoder_stages=None):
        """model: a WanVAE_ built by _video_vae_geoprior() (dual_branch=True or False)
        zmain_stats_path: json with the z_main statistics. None means z_main is not normalized.
        zprior_norm: normalize z_prior with the pretrained statistics.
                     True: when patchify was copy-initialized from the pretrained weights (nopatchify).
                     False: when loading a patchify trained with alignment (it was trained on raw).
        add_encoder_stages: [NEW 2026-09-04] the real stage list, used to compute the compression factors.
                     None keeps the old guess: 'one downsample3d if the parameter is present'."""
        super().__init__()
        self.model = model
        _temperal_downsample = [False, True, True]
        _dim_mult = [1, 2, 4, 4]
        add_enc = getattr(model, 'encoder', None)
        add_enc_stages = getattr(add_enc, 'add_downsamples', None)
        # [FIX 2026-09-04] this used to take no stage list and hardcode a guess of one downsample3d.
        #   That is right only for f16t8 (downsample3d x1); f32t4/c64 (downsample2d x2) came out 16/8
        #   instead of 32/4, so the latent grid and mask channels disagree and it silently decodes a different distribution.
        #   Use the real list when it is given, otherwise keep the old guess (existing callers unchanged).
        _stages = add_encoder_stages if add_encoder_stages else (
            [{"mode": "downsample3d"}]
            if (add_enc_stages and len(list(add_enc_stages.parameters())) > 0) else None)
        self.temporal_factor = _compute_temporal_factor(_temperal_downsample, _stages)
        self.upsampling_factor = _compute_spatial_factor(_dim_mult, _stages)
        self.z_dim = model.z_dim
        self.prior_z_dim = getattr(model, 'prior_z_dim', 16)
        # with dual_branch=False there is no prior, so latent_channels = z_dim only
        if getattr(model, 'dual_branch', False):
            self.latent_channels = model.z_dim + self.prior_z_dim  # for the DiT noise shape: z_main + z_prior
        else:
            self.latent_channels = model.z_dim

        # [new] z_prior scale normalization
        self._zprior_norm = zprior_norm
        if zprior_norm:
            _prior_mean = [-0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
                            0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921]
            _prior_std  = [ 2.8184,  1.4541,  2.3275,  2.6558,  1.2196,  1.7708,  2.6052,  2.0743,
                            3.2687,  2.1526,  2.8652,  1.5579,  1.6382,  1.1253,  2.8251,  1.9160]
            self.register_buffer('prior_mean',    torch.tensor(_prior_mean))
            self.register_buffer('prior_inv_std', torch.tensor([1.0 / s for s in _prior_std]))
            logging.info(f"[GeopriorVAE] z_prior normalization: ON (pretrained stats)")
        else:
            logging.info(f"[GeopriorVAE] z_prior normalization: OFF (raw)")

        # [modified] z_main normalization: loaded from json, or disabled
        self._zmain_norm = False
        if zmain_stats_path is not None:
            import json as _json
            with open(zmain_stats_path) as f:
                _stats = _json.load(f)
            self.register_buffer('main_mean', torch.tensor(_stats['mean']))
            self.register_buffer('main_inv_std', torch.tensor(_stats['inv_std']))
            self._zmain_norm = True
            logging.info(f"[GeopriorVAE] z_main normalization loaded from {zmain_stats_path}")
        else:
            logging.info(f"[GeopriorVAE] z_main normalization disabled (raw)")

    @property
    def _dtype(self):
        return next(self.model.parameters()).dtype

    def _encode_prior_full(self, video):
        """Run the prior encoder on the full-resolution input (same T'/H'/W' as z_main).
        _encode_prior() downsamples the input 2x before encoding, which causes a shape mismatch.
        At eval time the prior encoder is run directly, iterating the same way as the main encoder.
        """
        model = self.model
        tf = 1
        for td in model.temperal_downsample:
            if td:
                tf *= 2
        t = video.shape[2]
        prior_feat_map = [None] * model._cached_prior_conv_num
        iter_ = 1 + (t - 1) // tf
        out = None
        for i in range(iter_):
            prior_idx = [0]
            if i == 0:
                chunk = model.prior_encoder(video[:, :, :1, :, :],
                                            feat_cache=prior_feat_map, feat_idx=prior_idx)
                out = chunk
            else:
                chunk = model.prior_encoder(video[:, :, 1 + tf * (i - 1):1 + tf * i, :, :],
                                            feat_cache=prior_feat_map, feat_idx=prior_idx)
                out = torch.cat([out, chunk], dim=2)
        mu_prior, _ = model.prior_conv1(out).chunk(2, dim=1)
        # Apply same scale normalization as original Wan2.1 VAE
        mu_prior = (mu_prior - self.prior_mean.view(1, -1, 1, 1, 1).to(mu_prior)) * self.prior_inv_std.view(1, -1, 1, 1, 1).to(mu_prior)
        return mu_prior  # (1, prior_z_dim, T', H', W') — same T'/H'/W' as z_main

    def single_encode(self, video, device):
        """encode → z_main (dual_branch=False) or cat([z_main, z_prior]) (dual_branch=True).

        Note: _encode_prior() runs the prior encoder after a 2x avg_pool of the input.
        With a downsample3d in add_encoder_stages, so tf_main=8:
          z_main T' = 1+(17-1)//8 = 3
          _encode_prior: T_sub=9, tf_prior=4 -> T'_prior = 1+(9-1)//4 = 3  <- match
        Without add_encoder_stages, tf_main=4: T'=5 vs T'_prior=3 -> mismatch.
        """
        video = video.to(device=device, dtype=self._dtype)
        with torch.no_grad():
            mu_main, _ = self.model.encode(video, scale=None)                       # (1, z_dim, T', H', W')
            if self._zmain_norm:
                z_main = (mu_main - self.main_mean.view(1, -1, 1, 1, 1).to(mu_main)) * self.main_inv_std.view(1, -1, 1, 1, 1).to(mu_main)
            else:
                z_main = mu_main
            if getattr(self.model, 'dual_branch', False):
                z_prior = self.model._encode_prior(video)                           # (1, prior_z_dim, T', H', W')
                if self._zprior_norm:
                    z_prior = (z_prior - self.prior_mean.view(1, -1, 1, 1, 1).to(z_prior)) * self.prior_inv_std.view(1, -1, 1, 1, 1).to(z_prior)
                return torch.cat([z_main, z_prior], dim=1)                          # (1, z_dim+prior_z_dim, T', H', W')
        return z_main                                                               # (1, z_dim, T', H', W')

    def _denorm_latent(self, hidden_state, device):
        """[spatial-tile] shared by single and spatial-tiled decode: normalized latent -> model input space.
        Just the inverse-normalize of single_decode, split out (behaviour unchanged)."""
        hidden_state = hidden_state.to(device=device, dtype=self._dtype)
        if not getattr(self.model, 'dual_branch', False):
            return hidden_state[:, :self.model.z_dim]           # use z_main only
        z_main  = hidden_state[:, :self.z_dim]
        z_prior = hidden_state[:, self.z_dim:]
        if self._zmain_norm:
            z_main = z_main / self.main_inv_std.view(1, -1, 1, 1, 1).to(z_main) \
                     + self.main_mean.view(1, -1, 1, 1, 1).to(z_main)
        if self._zprior_norm:
            z_prior = z_prior / self.prior_inv_std.view(1, -1, 1, 1, 1).to(z_prior) \
                      + self.prior_mean.view(1, -1, 1, 1, 1).to(z_prior)
        return torch.cat([z_main, z_prior], dim=1)

    def single_decode(self, hidden_state, device):
        """hidden_state: z_main or cat([z_main, z_prior]) depending on dual_branch."""
        hidden_state = self._denorm_latent(hidden_state, device)
        video = self.model.decode(hidden_state, scale=None)
        return video.clamp_(-1, 1).float()

    # [spatial-tile] crossfade mask helpers - verbatim copies of the raw Wan class (lines 121-140)
    #   (the geoprior wrapper does not inherit that class, so it needs its own).
    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])
        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)
        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    # [spatial-tile] the first spatial tiled decode in the geoprior family.
    #   Note: decode(tiled=True) has always been silently ignored by this class (everything was single-pass),
    #   so to keep old results reproducible it fires only through the opt-in spatial_tiled argument (default False = no change at all).
    #   The crossfade blend of raw Wan tiled_decode (line 141), ported to geoprior (denorm + scale=None).
    def spatial_tiled_decode(self, hidden_state, device, tile_size=(20, 28), tile_stride=(10, 14)):
        hidden_state = self._denorm_latent(hidden_state, device)
        _, _, T, H, W = hidden_state.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        tasks = []
        for h in range(0, H, stride_h):
            if (h - stride_h >= 0 and h - stride_h + size_h >= H): continue
            for w in range(0, W, stride_w):
                if (w - stride_w >= 0 and w - stride_w + size_w >= W): continue
                tasks.append((h, min(h + size_h, H), w, min(w + size_w, W)))
        up = self.upsampling_factor
        out_T = T * self.temporal_factor - (self.temporal_factor - 1)
        weight = torch.zeros((1, 1, out_T, H * up, W * up), dtype=torch.float32, device="cpu")
        values = torch.zeros((1, 3, out_T, H * up, W * up), dtype=torch.float32, device="cpu")
        for h, h_, w, w_ in tasks:
            tile = hidden_state[:, :, :, h:h_, w:w_]
            video = self._decode_tile(tile, device, (h, h_, w, w_), (H, W)).float().cpu()
            mask = self.build_mask(video, is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                                   border_width=((size_h - stride_h) * up, (size_w - stride_w) * up)
                                   ).to(dtype=video.dtype, device="cpu")
            values[:, :, :, h * up:h * up + video.shape[3], w * up:w * up + video.shape[4]] += video * mask
            weight[:, :, :, h * up:h * up + video.shape[3], w * up:w * up + video.shape[4]] += mask
        values = values / weight
        return values.clamp_(-1, 1)

    def _decode_tile(self, tile, device, coords, full_hw):
        """[spatial-tile] decode one tile - the crossattn subclass overrides this to crop ff."""
        return self.model.decode(tile, scale=None)

    # [new] temporal tiled decode (the Open-Sora temporal_tiled_decode approach).
    #   The causal geoprior decoder accumulates drift over long sequences (81f) and the later frames collapse (per-frame 28 -> 14dB).
    #   Splitting the latent into overlapping windows along time and decoding each *independently* keeps every window short
    #   enough to stay in the "good regime" (the first 30f), which breaks the drift chain; a ramp overlap-add blend removes the seams. A decode-time fix, no retraining.
    #   Mapping: window[a:a+win] -> global frames [tf*a : tf*a + (1+(win-1)*tf)]  (tf=temporal_factor).
    def temporal_tiled_decode(self, hidden_state, device):
        win = int(getattr(self, 'temporal_tile_size', 5))      # latent window (frames = 1+(win-1)*tf)
        stride = int(getattr(self, 'temporal_tile_stride', 3))  # latent stride (win-stride = overlap)
        tf = int(self.temporal_factor)
        Tlat = hidden_state.shape[2]
        if Tlat <= win:
            return self.single_decode(hidden_state, device)
        starts = list(range(0, Tlat - win + 1, stride))
        if starts[-1] != Tlat - win:
            starts.append(Tlat - win)
        # [PORT from Bfix 2026-06-30] decode each window in a single pass (avoids the chunked frame drop; same as stage2 tiled_decode)
        _prev_fsp = getattr(self.model, 'force_single_pass', False)
        self.model.force_single_pass = True
        out = None; wsum = None
        for k, a in enumerate(starts):
            dec = self.single_decode(hidden_state[:, :, a:a + win], device)[0]  # (3, f, H, W)
            f = dec.shape[1]; g0 = tf * a
            if out is None:
                out_T = tf * (Tlat - 1) + 1
                out = torch.zeros(1, dec.shape[0], out_T, dec.shape[2], dec.shape[3], dtype=torch.float32)
                wsum = torch.zeros(out_T, dtype=torch.float32)
            g1 = min(g0 + f, out.shape[2]); ff = g1 - g0
            dec = dec[:, :ff].float().cpu()                          # (3, ff, H, W)
            bw = max(1, min(tf + 4, ff // 2))
            r = torch.ones(ff)
            if k != 0:                 r[:bw]  = torch.linspace(1.0 / bw, 1.0, bw)   # ramp up on the left
            if k != len(starts) - 1:   r[-bw:] = torch.linspace(1.0, 1.0 / bw, bw)   # ramp down on the right
            out[0, :, g0:g1] += dec * r.view(1, ff, 1, 1)
            wsum[g0:g1] += r
        out = out / wsum.clamp(min=1e-6).view(1, 1, -1, 1, 1)
        self.model.force_single_pass = _prev_fsp
        return out.clamp_(-1, 1)

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        videos = [v.to("cpu") for v in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            hidden_state = self.single_encode(video, device)
            hidden_states.append(hidden_state.squeeze(0))
        return torch.stack(hidden_states)

    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16),
               spatial_tiled=False, spatial_tile_size=(20, 28), spatial_tile_stride=(10, 14)):
        # [spatial-tile] the tiled argument has historically been ignored by this class (everything was
        #   single-pass). It stays that way to preserve reproducibility; spatial tiling is opt-in through spatial_tiled only.
        hidden_states = [h.to("cpu") for h in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if getattr(self, 'temporal_tile', False):
                video = self.temporal_tiled_decode(hidden_state, device)   # [NEW] long-video drift fix
            elif spatial_tiled:
                video = self.spatial_tiled_decode(hidden_state, device,
                                                  tile_size=spatial_tile_size,
                                                  tile_stride=spatial_tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            videos.append(video.squeeze(0))
        return torch.stack(videos)


def load_grace_geoprior_vae(
    checkpoint_path,
    pretrained_path,
    z_dim=32,
    prior_z_dim=16,
    add_encoder_stages=None,
    add_decoder_before_head_stages=None,
    no_expand_conv2=True,
    expand_encoder_head=False,
    no_dual_branch=False,
    device='cpu',
    dtype=torch.float32,
    zmain_stats_path=None,
    zprior_norm=True,
):
    """Load GRACEGeopriorVideoVAE from a geoprior training checkpoint.

    Parameters match train_causalvae_geoprior.py script arguments.
    Uses EMA weights when available.
    """
    logging.info(f"[GeopriorVAE] Building model (z_dim={z_dim}, prior_z_dim={prior_z_dim})")

    # Build model with pretrained init via _video_vae_geoprior
    model = _video_vae_geoprior(
        pretrained_path=pretrained_path,
        z_dim=z_dim,
        device='cpu',
        add_encoder_stages=add_encoder_stages,
        add_decoder_before_head_stages=add_decoder_before_head_stages,
        dual_branch=not no_dual_branch,
        subsample_mode='avg_pool',
        prior_z_dim=prior_z_dim,
        expand_conv2=not no_expand_conv2,
        expand_encoder_head=expand_encoder_head,
    )

    # Load training checkpoint (prefer EMA)
    if checkpoint_path is not None:
        logging.info(f"[GeopriorVAE] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # EMA weights preferred. When the EMA is nested ({shadow: params, shadow_buffers: buffers})
        # the two dicts are merged (REPA-E style); a flat one is used as is (legacy).
        # [FIX] loading the nested dict as-is made the top key 'shadow' look like a module name, so everything was silently missing.
        ema_sd = ckpt.get('ema_state_dict', {})
        if ema_sd:
            if isinstance(ema_sd, dict) and 'shadow' in ema_sd:
                _shadow = ema_sd.get('shadow', {})
                _shadow_buf = ema_sd.get('shadow_buffers', {})
                sd = {**_shadow, **_shadow_buf}
                logging.info(f"[GeopriorVAE] Using EMA weights (nested): shadow={len(_shadow)} + shadow_buffers={len(_shadow_buf)}")
            else:
                sd = ema_sd
                logging.info(f"[GeopriorVAE] Using EMA weights (flat): {len(sd)} keys")
        elif 'state_dict' in ckpt and 'gen_model' in ckpt['state_dict']:
            sd = ckpt['state_dict']['gen_model']
            logging.info(f"[GeopriorVAE] Using state_dict['gen_model']: {len(sd)} keys")
        else:
            sd = ckpt

        # Strip 'module.' prefix from DDP-wrapped checkpoints
        sd = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in sd.items()}
        # Strip 'vae.' prefix from GeopriorDiTAlignModel-wrapped checkpoints
        sd = {(k[len('vae.'):] if k.startswith('vae.') else k): v for k, v in sd.items()}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        logging.info(f"[GeopriorVAE] {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            logging.warning(f"[GeopriorVAE] Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            logging.warning(f"[GeopriorVAE] Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    vae = GRACEGeopriorVideoVAE(model, zmain_stats_path=zmain_stats_path, zprior_norm=zprior_norm)
    vae = vae.to(device=device, dtype=dtype).eval()
    return vae
