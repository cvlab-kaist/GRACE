#!/usr/bin/env bash
# One image-to-video sample with GRACE.
#   $1 is either one image or a folder of them; the script tells which by looking at the path.
#   $2 is the prompt; leave it out and the filename is used instead.
#
# ★ Every flag below was copied from the exact command that produced the paper's i2v
#   results (traced from the research runner). The architecture flags in particular are
#   load-bearing: drop --vae_r2n / --vae_no_expand_head and the VAE loads a different
#   structure; drop --rope_pos_scale and the DiT sees different positions. Either way you
#   get a plausible-looking video that is NOT what the paper reports.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN="${1:?usage: generate_i2v.sh <image|image_dir> [prompt] [out_dir]}"
PROMPT="${2:-}"
OUT="${3:-$HERE/outputs/i2v}"
# One file -> --image, a folder -> --image_dir. The argument order is unchanged, so older calls still work.
if [ -d "$IN" ]; then INPUT=(--image_dir "$IN"); else INPUT=(--image "$IN"); fi

GRACE_TASK=i2v . "$HERE/scripts/_resolve_paths.sh"   # fetch only the weights this task needs

: "${GRACE_WAN_I2V_DIR:?set GRACE_WAN_I2V_DIR to your Wan2.1-I2V-14B-480P folder (7 diffusion shards + google/umt5-xxl)}"
export GRACE_WAN_I2V_DIR

python "$HERE/src/inference_i2v_geoprior_crossattn.py" \
  --dit_checkpoint "${GRACE_DIT_I2V:?set GRACE_DIT_I2V (paper: step-3600.safetensors)}" \
  --lora_target_modules "${GRACE_LORA_TARGETS:-q,k,v,o,k_img,v_img,ffn.0,ffn.2}" \
  --vae_checkpoint "${GRACE_VAE_CKPT_I2V:?set GRACE_VAE_CKPT_I2V (paper: checkpoint-64000.ckpt)}" \
  --vae_pretrained "${GRACE_WAN_VAE:-$GRACE_WAN_I2V_DIR/Wan2.1_VAE.pth}" \
  --vae_z_dim 16 --vae_prior_z_dim 16 \
  --vae_r2n --vae_no_expand_head \
  --ff_window "${FF_WINDOW:-32}" --ff_encoder_source "${FF_ENC_SRC:-residual}" \
  ${GRACE_DECODER_CKPT_I2V:+--vae_decoder_checkpoint "$GRACE_DECODER_CKPT_I2V"} \
  --zmain_stats_path "${GRACE_ZMAIN_STATS_I2V:?set GRACE_ZMAIN_STATS_I2V}" \
  "${INPUT[@]}" --crop_input \
  ${PROMPT:+--prompt "$PROMPT"} \
  ${PROMPTS_JSON:+--prompts_json "$PROMPTS_JSON"} \
  --start_idx "${START:-0}" --n_videos "${N:--1}" \
  --height "${HEIGHT:-480}" --width "${WIDTH:-832}" --num_frames "${FRAMES:-81}" \
  --num_inference_steps "${STEPS:-50}" --cfg_scale "${CFG:-5.0}" \
  --async_delta "${GRACE_DELTA:-0.15}" --shift_main "${SHIFT_MAIN:-3}" --spatial_tile \
  --rope_pos_scale "${ROPE_SCALE:-2,2,2}" \
  --output_dir "$OUT" \
  ${NO_MERGE_LORA:+--no_merge_lora}
