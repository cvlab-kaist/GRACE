# train_crossattn

These four files define the **crossattn geoprior VAE architecture** and are copied
verbatim from the training checkout. They are required at inference time: the VAE
builder loads `grace_geoprior.py` from here by absolute path.

Do not replace them with the older copies in `src/` — those predate
`stages_norm_before_head` and will not load the released checkpoints.
