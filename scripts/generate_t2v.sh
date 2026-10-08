#!/usr/bin/env bash
# One text-to-video sample with GRACE.
#   Every flag below matches the command that produced the paper's T2V results. The ones with
#   defaults (z dims, LoRA targets/rank, rope scale) are spelled out on purpose: they must match
#   training, and a wrong value yields a plausible video that is not what the paper reports.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT="${1:?usage: generate_t2v.sh <prompt> [out_dir]}"
OUT="${2:-$HERE/outputs/t2v}"

GRACE_TASK=t2v . "$HERE/scripts/_resolve_paths.sh"   # fetch only the weights this task needs

: "${GRACE_WAN_T2V_DIR:?set GRACE_WAN_T2V_DIR to your Wan2.1-T2V-14B folder}"
: "${GRACE_WAN_SHARED:?set GRACE_WAN_SHARED to your Wan2.1-I2V-14B-480P folder (T5, Wan2.1_VAE.pth, google/umt5-xxl)}"

python "$HERE/src/inference_t2v_geoprior.py" \
  --prompt "$PROMPT" --out_dir "$OUT" \
  --dit_checkpoint "${GRACE_DIT_T2V:?set GRACE_DIT_T2V (paper: step-4125.safetensors)}" \
  --dit_dir "$GRACE_WAN_T2V_DIR" --shared_dir "$GRACE_WAN_SHARED" \
  --vae_checkpoint "${GRACE_VAE_CKPT:?set GRACE_VAE_CKPT (paper: checkpoint-29000.ckpt)}" \
  ${GRACE_DECODER_CKPT:+--vae_decoder_checkpoint "$GRACE_DECODER_CKPT"} \
  --zmain_stats_path "${GRACE_ZMAIN_STATS:?set GRACE_ZMAIN_STATS}" \
  --vae_z_dim 16 --vae_prior_z_dim 16 --vae_prior_subsample_mode bilinear \
  --lora_target_modules "${GRACE_LORA_TARGETS:-q,k,v,o,ffn.0,ffn.2}" --lora_rank "${GRACE_LORA_RANK:-512}" \
  --rope_pos_scale "${ROPE_SCALE:-2,2,2}" \
  --height "${HEIGHT:-480}" --width "${WIDTH:-832}" --num_frames "${FRAMES:-81}" \
  --num_inference_steps "${STEPS:-50}" --cfg_scale "${CFG:-5.0}" \
  --async_delta "${GRACE_DELTA:-0.15}" --seed "${SEED:-0}" --fps "${FPS:-16}" \
  ${LOSSLESS:+--lossless} \
  ${NO_MERGE_LORA:+--no_merge_lora}
