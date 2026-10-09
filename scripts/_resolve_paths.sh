# Shared path resolution, sourced by generate_*.sh.
#   One variable is enough:   export GRACE_CKPT_DIR=./checkpoints
# It expects the layout of our HuggingFace release:
#   checkpoints/
#     t2v/{dit.safetensors, vae.ckpt, decoder.ckpt, zmain_stats.json}
#     i2v/{dit.safetensors, vae.ckpt, decoder.ckpt, zmain_stats.json}
# Any individual GRACE_* variable still wins, so a file can live anywhere.
# No GRACE_CKPT_DIR? Fall back to ./checkpoints, and fetch it on first use.
#   Auto-download is opt-out (GRACE_NO_AUTO_DOWNLOAD=1) rather than opt-in, so a fresh
#   clone runs with one command. It only pulls the task you are about to run.
# generate_*.sh already set HERE; fall back to this file's own location so the README's
# `. scripts/_resolve_paths.sh` works on its own too.
HERE="${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
_d="${GRACE_CKPT_DIR:-$HERE/checkpoints}"
if [ ! -d "$_d" ] && [ -z "${GRACE_NO_AUTO_DOWNLOAD:-}" ]; then
  echo "[grace] checkpoints not found at $_d - downloading from ${GRACE_HF_REPO:-chimaharicox/GRACE}"
  python "$HERE/tools/download_weights.py" --task "${GRACE_TASK:-both}" --dest "$_d" || {
    echo "[grace] download failed. Fetch manually: python tools/download_weights.py"; exit 1; }
fi
if [ -n "$_d" ]; then
  : "${GRACE_DIT_T2V:=$_d/t2v/dit.safetensors}"
  : "${GRACE_VAE_CKPT:=$_d/t2v/vae.ckpt}"
  : "${GRACE_DECODER_CKPT:=$_d/t2v/decoder.ckpt}"
  : "${GRACE_ZMAIN_STATS:=$_d/t2v/zmain_stats.json}"
  : "${GRACE_DIT_I2V:=$_d/i2v/dit.safetensors}"
  : "${GRACE_VAE_CKPT_I2V:=$_d/i2v/vae.ckpt}"
  : "${GRACE_DECODER_CKPT_I2V:=$_d/i2v/decoder.ckpt}"
  : "${GRACE_ZMAIN_STATS_I2V:=$_d/i2v/zmain_stats.json}"
  # I2V is the one task with resolution-specific weights: a separate DiT and separate latent
  # statistics for 736x1280. generate_i2v.sh switches to these when HEIGHT is 736 or more.
  # T2V needs no equivalent - one checkpoint covers both resolutions.
  : "${GRACE_DIT_I2V_736:=$_d/i2v/dit_736.safetensors}"
  : "${GRACE_ZMAIN_STATS_I2V_736:=$_d/i2v/zmain_stats_736.json}"
  export GRACE_DIT_T2V GRACE_VAE_CKPT GRACE_DECODER_CKPT GRACE_ZMAIN_STATS
  export GRACE_DIT_I2V GRACE_VAE_CKPT_I2V GRACE_DECODER_CKPT_I2V GRACE_ZMAIN_STATS_I2V
  export GRACE_DIT_I2V_736 GRACE_ZMAIN_STATS_I2V_736
fi
# The Wan2.1 base weights are downloaded separately, so they keep their own variables.
# GRACE_WAN_SHARED defaults to the I2V folder: T2V needs the T5 / VAE / tokenizer that live there.
: "${GRACE_WAN_SHARED:=${GRACE_WAN_I2V_DIR:-}}"
export GRACE_WAN_SHARED
