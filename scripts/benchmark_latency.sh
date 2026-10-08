#!/usr/bin/env bash
# Reproduce the latency reported in the paper, with this repository's code.
#
# The measured window is the one every model in the paper's table was measured over:
#   t2v core = first DiT forward of the denoising loop -> end of the final VAE decode
#   i2v core = start of the first VAE encode           -> end of the final VAE decode
# Checkpoint loading, text encoding and mp4 writing are outside it.
#
# The two settings that matter are the defaults of the entry points and are set
# explicitly here so the command documents itself:
#   LoRA folded into the base weights, and the DiT kept on the GPU during decode.
# Turning either off costs time without changing what the model computes.
#
#   usage: bash scripts/benchmark_latency.sh [t2v|i2v]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-t2v}"
OUT="${OUT:-$HERE/outputs/latency_$TASK}"
RESULT="${RESULT:-$HERE/outputs/latency_$TASK.json}"

GRACE_TASK="$TASK" . "$HERE/scripts/_resolve_paths.sh"
export KINEMA_MERGE_LORA="${KINEMA_MERGE_LORA:-1}"
export DIT_OFFLOAD_DECODE="${DIT_OFFLOAD_DECODE:-0}"
mkdir -p "$(dirname "$RESULT")"; rm -rf "$OUT"

if [ "$TASK" = "t2v" ]; then
  : "${GRACE_WAN_T2V_DIR:?set GRACE_WAN_T2V_DIR to your Wan2.1-T2V-14B folder}"
  : "${GRACE_WAN_SHARED:?set GRACE_WAN_SHARED to your Wan2.1-I2V-14B-480P folder}"
  exec python "$HERE/tools/benchmark_latency.py" --task t2v --out "$RESULT" -- \
    --prompt "${PROMPTS:-$HERE/assets/bench_prompts.txt}" --out_dir "$OUT" \
    --dit_checkpoint "${GRACE_DIT_T2V:?set GRACE_DIT_T2V}" \
    --dit_dir "$GRACE_WAN_T2V_DIR" --shared_dir "$GRACE_WAN_SHARED" \
    --vae_checkpoint "${GRACE_VAE_CKPT:?set GRACE_VAE_CKPT}" \
    ${GRACE_DECODER_CKPT:+--vae_decoder_checkpoint "$GRACE_DECODER_CKPT"} \
    --zmain_stats_path "${GRACE_ZMAIN_STATS:?set GRACE_ZMAIN_STATS}" \
    --vae_z_dim 16 --vae_prior_z_dim 16 --vae_prior_subsample_mode bilinear \
    --lora_target_modules "${GRACE_LORA_TARGETS:-q,k,v,o,ffn.0,ffn.2}" --lora_rank "${GRACE_LORA_RANK:-512}" \
    --rope_pos_scale "${ROPE_SCALE:-2,2,2}" \
    --height "${HEIGHT:-480}" --width "${WIDTH:-832}" --num_frames "${FRAMES:-81}" \
    --num_inference_steps "${STEPS:-50}" --cfg_scale "${CFG:-5.0}" \
    --async_delta "${GRACE_DELTA:-0.15}" --shift_main "${SHIFT_MAIN:-3}" --seed "${SEED:-0}" --fps 16
fi

: "${GRACE_WAN_I2V_DIR:?set GRACE_WAN_I2V_DIR to your Wan2.1-I2V-14B-480P folder}"
exec python "$HERE/tools/benchmark_latency.py" --task i2v --out "$RESULT" -- \
  --dit_checkpoint "${GRACE_DIT_I2V:?set GRACE_DIT_I2V}" \
  --dit_dir "$GRACE_WAN_I2V_DIR" --shared_dir "$GRACE_WAN_I2V_DIR" \
  --vae_checkpoint "${GRACE_VAE_CKPT_I2V:?set GRACE_VAE_CKPT_I2V}" \
  --vae_pretrained "${GRACE_WAN_VAE:-$GRACE_WAN_I2V_DIR/Wan2.1_VAE.pth}" \
  --vae_z_dim 16 --vae_prior_z_dim 16 --vae_r2n --vae_no_expand_head \
  --ff_window "${FF_WINDOW:-32}" --ff_encoder_source "${FF_ENC_SRC:-residual}" \
  ${GRACE_DECODER_CKPT_I2V:+--vae_decoder_checkpoint "$GRACE_DECODER_CKPT_I2V"} \
  --zmain_stats_path "${GRACE_ZMAIN_STATS_I2V:?set GRACE_ZMAIN_STATS_I2V}" \
  --image_dir "${IMGDIR:?set IMGDIR to a folder of .jpg inputs}" --crop_input \
  --n_videos "${N:-5}" --start_idx 0 \
  --height "${HEIGHT:-480}" --width "${WIDTH:-832}" --num_frames "${FRAMES:-81}" \
  --num_inference_steps "${STEPS:-50}" --cfg_scale "${CFG:-5.0}" \
  --async_delta "${GRACE_DELTA:-0.15}" --shift_main "${SHIFT_MAIN:-3}" --spatial_tile \
  --rope_pos_scale "${ROPE_SCALE:-2,2,2}" --output_dir "$OUT"
