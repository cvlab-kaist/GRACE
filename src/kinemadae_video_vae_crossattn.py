# [crossattn-inference] VAE loader and decoder compatible with first-frame cross-attention.
#
# Purpose: take a VAE trained with first_frame_inject=True and swap in **only its decoder** for
#       inference. kinemadae_video_vae.py is left untouched.
#
# Why a separate file:
#   - The older modules/kinemadae_geoprior.py in the inference tree has no crossattn architecture
#     (ff_inject, GatedCrossAttn, first_frame) at all, so loading a crossattn checkpoint would drop
#     the weights silently under strict=False - the same trap as the swirl-artifact incident.
#   - So the crossattn architecture is imported from the *training* repo, which has WanVAE_ and
#     crossattn_ff.GatedCrossAttnBlock, making it bit-identical to training.
#
# The encode, normalize and temporal logic of the existing wrapper (KinemaDAEGeopriorVideoVAE) is
# inherited unchanged; only the decode path is overridden to pass the first frame through

import logging
import os
import sys

import torch

# ---------------------------------------------------------------------------
# The crossattn architecture lives in the training repo's kinemadae_geoprior (its own WanVAE_ plus
# a crossattn_ff import), so that repo has to be on sys.path for that import to resolve.
# ---------------------------------------------------------------------------
# [release] The default is the copy bundled in this repo. The entry points setdefault the same value,
#   but this module also points at it so that importing it directly works.
_TRAIN_CROSSATTN_REPO = os.environ.get(
    "KINEMADAE_CROSSATTN_REPO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_crossattn"))
if _TRAIN_CROSSATTN_REPO not in sys.path:
    sys.path.insert(0, _TRAIN_CROSSATTN_REPO)

# the training repo's builder, which supports WanVAE_(first_frame_inject=...).
# NOTE: the inference tree has a module of the same name, so it is loaded by absolute path with
#       importlib to remove any dependence on sys.path order.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "kinemadae_geoprior_crossattn_train",
    os.path.join(_TRAIN_CROSSATTN_REPO, "kinemadae_geoprior.py"),
)
_kg_train = _ilu.module_from_spec(_spec)
sys.modules["kinemadae_geoprior_crossattn_train"] = _kg_train
_spec.loader.exec_module(_kg_train)
_video_vae_geoprior_crossattn = _kg_train._video_vae_geoprior

# reuse the existing inference wrapper (encode, single_encode, temporal_tiled_decode, normalize)
from kinemadae_video_vae import KinemaDAEGeopriorVideoVAE  # noqa: E402


class KinemaDAEGeopriorVideoVAECrossAttn(KinemaDAEGeopriorVideoVAE):
    """Same as the parent (KinemaDAEGeopriorVideoVAE) but passes the first frame's pixels into decode.

    single_decode: identical to the parent's inverse-normalize logic, with first_frame= added to the
                   final self.model.decode call. crossattn was only verified single-pass during
                   training (force_single_pass), so temporal tiling is not used here.
    """

    def single_decode(self, hidden_state, device, first_frame=None):
        # --- the inverse-normalize block below is identical to the parent's single_decode ---
        hidden_state = hidden_state.to(device=device, dtype=self._dtype)
        if not getattr(self.model, 'dual_branch', False):
            hidden_state = hidden_state[:, :self.model.z_dim]   # use z_main only
        else:
            z_main = hidden_state[:, :self.z_dim]
            z_prior = hidden_state[:, self.z_dim:]
            if self._zmain_norm:
                z_main = z_main / self.main_inv_std.view(1, -1, 1, 1, 1).to(z_main) \
                    + self.main_mean.view(1, -1, 1, 1, 1).to(z_main)
            if self._zprior_norm:
                z_prior = z_prior / self.prior_inv_std.view(1, -1, 1, 1, 1).to(z_prior) \
                    + self.prior_mean.view(1, -1, 1, 1, 1).to(z_prior)
            hidden_state = torch.cat([z_main, z_prior], dim=1)
        # --- the only difference: pass first_frame (None behaves exactly like the parent) ---
        _ff = None
        if first_frame is not None:
            _ff = first_frame.to(device=device, dtype=self._dtype)
        video = self.model.decode(hidden_state, scale=None, first_frame=_ff)
        return video.clamp_(-1, 1).float()

    # [spatial-tile] Override the parent's per-tile decode: convert the tile's latent coordinates to
    #   pixel coordinates (x upsampling_factor) and crop first_frame to the same region.
    #   The model builds the ff pyramid from the cropped pixels, so window alignment holds inside the tile.
    #   Truncating the ff encoder's receptive field at tile borders is absorbed by the overlap and crossfade.
    def _decode_tile(self, tile, device, coords, full_hw):
        _ff = None
        if getattr(self, "_ff_current", None) is not None:
            h, h_, w, w_ = coords
            up = self.upsampling_factor
            _ff = self._ff_current[:, :, h * up:h_ * up, w * up:w_ * up].to(device=device, dtype=self._dtype)
        return self.model.decode(tile, scale=None, first_frame=_ff)

    def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34),
               tile_stride=(18, 16), first_frame=None,
               spatial_tiled=False, spatial_tile_size=(20, 28), spatial_tile_stride=(10, 14)):
        """first_frame: (1,3,H,W) first-frame pixels in [-1,1] at full resolution, used as the
        cross-attention K/V features.
        Single pass by default. With spatial_tiled=True it decodes per tile, cropping the first frame
        per tile, which lowers peak memory."""
        hidden_states = [h.to("cpu") for h in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if spatial_tiled:
                self._ff_current = first_frame          # _decode_tile crops per coordinate
                try:
                    video = self.spatial_tiled_decode(hidden_state, device,
                                                      tile_size=spatial_tile_size,
                                                      tile_stride=spatial_tile_stride)
                finally:
                    self._ff_current = None
            else:
                video = self.single_decode(hidden_state, device, first_frame=first_frame)
            videos.append(video.squeeze(0))
        return torch.stack(videos)


def load_kinemadae_geoprior_vae_crossattn(
    checkpoint_path,
    pretrained_path,
    z_dim=16,
    prior_z_dim=16,
    add_encoder_stages=None,
    add_decoder_before_head_stages=None,
    # [f32t4 / c64] this lineage uses add_decoder_stages (upsample2d_keepdim x2) rather than before_head.
    #   With None the key is not passed at all, so the old behaviour is unchanged.
    add_decoder_stages=None,
    no_expand_conv2=True,
    subsample_mode='avg_pool',          # encode stays identical to the existing inference path, so the DiT conditioning and latents are unchanged
    first_frame_inject=True,
    ff_window=32,
    ff_encoder_source='residual',
    ff_inject_levels='all',
    ff_dual_source=False,   # [dual] for checkpoints carrying both the residual and base references
    # [R2n / ff] architecture arguments. The defaults leave every existing path unchanged.
    #   KINEMADAE_CROSSATTN_REPO must point at a training repo that supports R2n for the builder to read them.
    stages_after_norm=False,
    stages_norm_before_head=False,
    expand_encoder_head=False,
    decoder_mirror=True,
    use_b_adaptive=False,
    b_adaptive_max=100000,
    # [decoder-swap] base (encoder and prior) comes from checkpoint_path; only the trained decoder and ff
    #   are overlaid from this checkpoint. Trained keys are identified by comparing raw tensors bit for bit
    #   (frozen keys equal the base raw, only trained ones differ), so this is for decoder-only checkpoints.
    decoder_checkpoint_path=None,
    device='cpu',
    dtype=torch.float32,
    zmain_stats_path=None,
    zprior_norm=True,
):
    """Load a crossattn (first_frame_inject) VAE. Same flow as load_kinemadae_geoprior_vae, plus the
    first_frame_inject and ff_* build arguments, the training repo's builder, and a forced single pass.
    """
    logging.info(f"[GeopriorVAE-crossattn] Building model (z_dim={z_dim}, prior_z_dim={prior_z_dim}, "
                 f"first_frame_inject={first_frame_inject}, ff_window={ff_window}, "
                 f"ff_encoder_source={ff_encoder_source})")

    model = _video_vae_geoprior_crossattn(
        pretrained_path=pretrained_path,
        z_dim=z_dim,
        device='cpu',
        add_encoder_stages=add_encoder_stages,
        add_decoder_before_head_stages=add_decoder_before_head_stages,
        # [f32t4 / c64] with None the key itself is not passed, which keeps older builders working
        **({"add_decoder_stages": add_decoder_stages} if add_decoder_stages else {}),
        dual_branch=True,
        subsample_mode=subsample_mode,
        prior_z_dim=prior_z_dim,
        expand_conv2=not no_expand_conv2,
        # --- crossattn build arguments (the training builder forwards them to WanVAE_ as **kwargs) ---
        first_frame_inject=first_frame_inject,
        ff_window=ff_window,
        ff_encoder_source=ff_encoder_source,
        ff_inject_levels=ff_inject_levels,
        ff_dual_source=ff_dual_source,   # [dual]
        # [R2n] only passed when non-default - the older builder does not know these kwargs and would
        #   raise TypeError. The default path passes exactly what it did before.
        **({"stages_after_norm": stages_after_norm,
            "stages_norm_before_head": stages_norm_before_head,
            "expand_encoder_head": expand_encoder_head,
            "decoder_mirror": decoder_mirror,
            "use_b_adaptive": use_b_adaptive,
            "b_adaptive_max": b_adaptive_max}
           if (stages_after_norm or stages_norm_before_head or expand_encoder_head
               or not decoder_mirror or use_b_adaptive) else {}),
    )
    # crossattn was only verified single-pass during training, so decoding stays single-pass even in eval.
    model.force_single_pass = True

    if checkpoint_path is not None:
        logging.info(f"[GeopriorVAE-crossattn] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # prefer EMA (nested {shadow, shadow_buffers}), the same logic as the existing loader.
        ema_sd = ckpt.get('ema_state_dict', {})
        if ema_sd:
            if isinstance(ema_sd, dict) and 'shadow' in ema_sd:
                _shadow = ema_sd.get('shadow', {})
                _shadow_buf = ema_sd.get('shadow_buffers', {})
                sd = {**_shadow, **_shadow_buf}
                logging.info(f"[GeopriorVAE-crossattn] Using EMA (nested): shadow={len(_shadow)} + buffers={len(_shadow_buf)}")
            else:
                sd = ema_sd
                logging.info(f"[GeopriorVAE-crossattn] Using EMA (flat): {len(sd)} keys")
        elif 'state_dict' in ckpt and 'gen_model' in ckpt['state_dict']:
            sd = ckpt['state_dict']['gen_model']
            logging.info(f"[GeopriorVAE-crossattn] Using state_dict['gen_model']: {len(sd)} keys")
        else:
            sd = ckpt

        # strip the DDP 'module.' and align-wrapper 'vae.' prefixes to line up with WanVAE_ keys
        sd = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in sd.items()}
        sd = {(k[len('vae.'):] if k.startswith('vae.') else k): v for k, v in sd.items()}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        # crossattn check: confirm the ff_inject and gamma keys really loaded (leaking into unexpected means an architecture mismatch)
        _ff_model = [k for k in model.state_dict() if 'ff_inject' in k]
        _ff_unexp = [k for k in unexpected if 'ff_inject' in k]
        _ff_miss = [k for k in missing if 'ff_inject' in k]
        logging.info(f"[GeopriorVAE-crossattn] load: {len(missing)} missing, {len(unexpected)} unexpected "
                     f"| ff_inject: model={len(_ff_model)} missing={len(_ff_miss)} unexpected={len(_ff_unexp)}")
        if _ff_miss and decoder_checkpoint_path is None:
            logging.error(f"[GeopriorVAE-crossattn] ff_inject MISSING - crossattn weights were not loaded: {_ff_miss[:4]}")
        if _ff_unexp:
            logging.error(f"[GeopriorVAE-crossattn] ff_inject UNEXPECTED (arch mismatch!): {_ff_unexp[:4]}")
        if unexpected:
            logging.warning(f"[GeopriorVAE-crossattn] unexpected (non-ff): {[k for k in unexpected if 'ff_inject' not in k][:5]}")

        # [decoder-swap] overlay the trained decoder and ff, identified by comparing base raw against donor raw
        if decoder_checkpoint_path is not None:
            logging.info(f"[GeopriorVAE-crossattn] decoder-swap from {decoder_checkpoint_path}")
            _dn = torch.load(decoder_checkpoint_path, map_location='cpu', weights_only=False)
            def _normkeys(sdd):
                sdd = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in sdd.items()}
                return {(k[len('vae.'):] if k.startswith('vae.') else k): v for k, v in sdd.items()}
            donor_raw = _normkeys(_dn['state_dict']['gen_model'])
            _de = _dn.get('ema_state_dict', {})
            if isinstance(_de, dict) and 'shadow' in _de:
                donor_ema = _normkeys({**_de.get('shadow', {}), **_de.get('shadow_buffers', {})})
            elif _de:
                donor_ema = _normkeys(_de)
            else:
                donor_ema = donor_raw
            base_raw = _normkeys(ckpt['state_dict']['gen_model']) if ('state_dict' in ckpt and 'gen_model' in ckpt['state_dict']) else {}
            own = model.state_dict()
            overlay = {}
            for k, v_raw in donor_raw.items():
                if k not in own or own[k].shape != v_raw.shape:
                    continue
                _b = base_raw.get(k)
                if _b is None or _b.shape != v_raw.shape or not torch.equal(_b, v_raw):
                    overlay[k] = donor_ema.get(k, v_raw)   # only the trained keys (or ff keys absent from base)
            _m2, _u2 = model.load_state_dict(overlay, strict=False)
            _ff_ol = sum(1 for k in overlay if 'ff_inject' in k)
            logging.info(f"[GeopriorVAE-crossattn] decoder-swap overlay: {len(overlay)} keys "
                         f"({_ff_ol} of them ff_inject), overlay-unexpected={len(_u2)}")
            _ff_still = [k for k in model.state_dict() if 'ff_inject' in k and k not in overlay]
            if _ff_still:
                logging.error(f"[GeopriorVAE-crossattn] ff still not loaded after the decoder swap: {len(_ff_still)} keys {_ff_still[:3]}")

    # Pass the actual encoder stage list. Without it the parent assumes a single downsample3d and gets
    #   the compression factor wrong for f32t4 (c64, downsample2d x2) - 16/8 instead of 32/4.
    vae = KinemaDAEGeopriorVideoVAECrossAttn(model, zmain_stats_path=zmain_stats_path,
                                             zprior_norm=zprior_norm,
                                             add_encoder_stages=add_encoder_stages)
    vae = vae.to(device=device, dtype=dtype).eval()
    return vae
