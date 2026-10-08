import torch


class DaVaeHeadDual(torch.nn.Module):
    """Dual-branch head: head_prior may be conditioned on its own timestep embedding.

    Lifecycle: `t_prior_emb` is a plain attribute the sampling loop sets each step and
    resets to None when done. It intentionally PERSISTS across calls (so the CFG
    positive/negative passes within one step share it) — do not auto-clear. When set it is
    cast to `mod`'s dtype/device inside forward, so a float32 embedding is safe even when the
    head runs in bf16. When None, forward is identical to the original DaVaeHead (so beta
    equality reproduces the baseline)."""
    def __init__(self, head_main, head_prior):
        super().__init__()
        self.head_main = head_main
        self.head_prior = head_prior
        self.t_prior_emb = None   # Optional[torch.Tensor]; caller-managed (see docstring)

    def forward(self, x, mod):
        mod_prior = self.t_prior_emb.to(mod) if self.t_prior_emb is not None else mod
        return torch.cat([self.head_main(x, mod), self.head_prior(x, mod_prior)], dim=-1)
