import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 1e-6

def atanh(x):
    return 0.5 * (torch.log1p(x + EPS) - torch.log1p(-x + EPS))

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init,
                 a_min=-3.0, a_max=+2.0, device=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.register_buffer("a_min", torch.tensor(a_min, dtype=torch.float32))
        self.register_buffer("a_max", torch.tensor(a_max, dtype=torch.float32))
        self.register_buffer("a_range", self.a_max - self.a_min)
        self.device = torch.device("cpu") if device is None else torch.device(device)

        # ------- Actor (mean head only; no final activation) -------
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 500), nn.Tanh(),
            nn.Linear(500, 300),          nn.Tanh(),
            nn.Linear(300, 100),          nn.Tanh(),
            nn.Linear(100, self.action_dim)  # <- no activation here
        )
        # log-std as a learnable parameter (shared across states; per-dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), np.log(action_std_init), dtype=torch.float32))

        # ------- Critic -------
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 500), nn.Tanh(),
            nn.Linear(500, 300),          nn.Tanh(),
            nn.Linear(300, 100),          nn.Tanh(),
            nn.Linear(100, 1)
        )

        self.to(self.device)

        # constant Jacobian term for scaling u∈[-1,1] → a∈[a_min,a_max]
        # du/da = 2 / (a_max - a_min)
        self.register_buffer("log_du_da_const",
            torch.sum(torch.log(2.0 / (self.a_range + EPS))))

    # ------------ Utilities ------------
    def _ensure_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # (B=1, ...)
        return x

    def set_action_std(self, new_action_std):
        with torch.no_grad():
            self.log_std.copy_(torch.full_like(self.log_std, np.log(new_action_std), dtype=torch.float32))

    def get_value(self, obs):
        obs = self._ensure_tensor(obs)
        return self.critic(obs)

    # ------------ Sampling (for rollout) ------------
    def get_action_and_log_prob(self, obs):
        """
        Returns:
          action: (B, action_dim) in physical units [a_min, a_max], detached
          logp:   (B,) log-prob of *that* action under current policy
        """
        obs = self._ensure_tensor(obs)
        mu = self.actor(obs)                            # (B, A), unbounded
        std = self.log_std.exp()                        # (A,)
        std = std.expand_as(mu)                         # (B, A)

        # sample in R^A, then tanh-squash to [-1,1]
        eps = torch.randn_like(mu)
        pre = mu + std * eps                            # (B, A)
        u = torch.tanh(pre)                             # (B, A) in [-1,1]

        # scale to [a_min, a_max]
        a = self.a_min + (u + 1.0) * 0.5 * self.a_range  # (B, A)

        # log-prob with tanh correction + scaling Jacobian
        # log N(pre; mu, std)  minus  sum log(1 - tanh(pre)^2)
        logp_pre = (-0.5 * (((pre - mu) / (std + EPS))**2 + 2*torch.log(std + EPS) + np.log(2*np.pi))).sum(dim=-1)
        logp_tanh = torch.log(1.0 - u.pow(2) + EPS).sum(dim=-1)  # this is negative
        # scaling u->a adds a constant per-dim term: sum log(du/da)
        logp = logp_pre - logp_tanh + self.log_du_da_const

        return a.detach().cpu().numpy(), logp.detach()

    # ------------ Evaluate (for PPO update) ------------
    def evaluate(self, obs, action):
        """
        Inputs:
          obs:    (B, obs_dim)
          action: (B, action_dim) in physical units [a_min, a_max]
        Returns:
          logp:   (B,)
          values: (B,1)
          entropy:(B,)
        """
        obs = self._ensure_tensor(obs)
        action = self._ensure_tensor(action)

        # map physical action a -> u in (-1,1) and pre = atanh(u)
        # u = 2*(a - a_min)/range - 1
        u = 2.0 * (action - self.a_min) / (self.a_range + EPS) - 1.0
        u = torch.clamp(u, -1.0 + 1e-6, 1.0 - 1e-6)
        pre = atanh(u)

        mu = self.actor(obs)                  # (B, A)
        std = self.log_std.exp().expand_as(mu)
        # log N(pre; mu, std)
        logp_pre = (-0.5 * (((pre - mu) / (std + EPS))**2 + 2*torch.log(std + EPS) + np.log(2*np.pi))).sum(dim=-1)
        # tanh correction and scaling Jacobian
        logp_tanh = torch.log(1.0 - u.pow(2) + EPS).sum(dim=-1)
        logp = logp_pre - logp_tanh + self.log_du_da_const

        # entropy of the *pre-squash* Gaussian (common in PPO; stable proxy)
        entropy = (0.5 + 0.5*np.log(2*np.pi) + torch.log(std + EPS)).sum(dim=-1)

        values = self.critic(obs)
        return logp, values, entropy