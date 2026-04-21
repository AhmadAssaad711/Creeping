import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime 

from .ppo import ActorCritic
from ..parameters import *
from ..simulation.settings import MAX_ACCEL, MAX_DECEL


device = torch.device("cpu")

def pick_device(use_cuda: bool = True):
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class Buffer:
    def __init__(self):
        self.observation = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.observation.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

class PPOAgent(object):
    def __init__(self, town, action_std_init=0.4, use_cuda=True,
                 a_min=None, a_max=None, obs_dim=107, action_dim=1):

        self.device = pick_device(use_cuda)

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)         # acceleration only
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = 7
        self.lr = PPO_LEARNING_RATE
        self.action_std = float(action_std_init)
        self.memory = Buffer()
        self.town = town

        # Use env-consistent accel bounds if you have them in parameters.py
        # If you don't, keep None and ActorCritic defaults [-3,+2] will apply.
        if a_min is None and "MAX_DECEL" in globals():
            a_min = -float(MAX_DECEL)  # should be negative
        if a_max is None and "MAX_ACCEL" in globals():
            a_max = float(MAX_ACCEL)

        ac_kwargs = dict(obs_dim=self.obs_dim, action_dim=self.action_dim,
                         action_std_init=self.action_std, device=self.device)
        if a_min is not None and a_max is not None:
            ac_kwargs["a_min"] = a_min
            ac_kwargs["a_max"] = a_max

        self.policy = ActorCritic(**ac_kwargs)
        self.old_policy = ActorCritic(**ac_kwargs)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(), "lr": self.lr},
            {"params": self.policy.critic.parameters(), "lr": self.lr},
            {"params": [self.policy.log_std], "lr": self.lr},
        ])

        self.MseLoss = nn.MSELoss()

        # optional: for your checkpoint metadata
        self.runtime_progress = {}

    def get_action(self, obs, train: bool):
        """
        Returns: accel_cmd as a python float in [a_min, a_max] (continuous).
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            action_np, logprob = self.old_policy.get_action_and_log_prob(obs_t)
            # action_np is (B=1, A=1) numpy -> scalar float
            accel_cmd = float(np.asarray(action_np).squeeze())

        if train:
            self.memory.observation.append(obs_t.detach())
            self.memory.actions.append(torch.tensor([accel_cmd], dtype=torch.float32, device=self.device))
            self.memory.log_probs.append(logprob.squeeze(0).detach())

        return accel_cmd

    def set_action_std(self, new_action_std):
        self.action_std = float(new_action_std)
        self.policy.set_action_std(self.action_std)
        self.old_policy.set_action_std(self.action_std)

    def learn(self):
        # 1) Monte-Carlo returns
        rewards = []
        discounted_reward = 0.0
        for r, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if done:
                discounted_reward = 0.0
            discounted_reward = float(r) + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 2) Stack rollout tensors
        old_states   = torch.stack(self.memory.observation, dim=0)     # (T, obs_dim)
        old_actions  = torch.stack(self.memory.actions, dim=0)         # (T, 1)
        old_logprobs = torch.stack(self.memory.log_probs, dim=0)       # (T,)

        # 3) PPO update
        for _ in range(self.n_updates_per_iteration):
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            values = values.squeeze(-1)  # (T,)

            advantages = rewards - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = self.MseLoss(values, rewards)
            entropy_bonus = dist_entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    @staticmethod
    def _now():
        return datetime.now().isoformat(timespec="seconds")

    def chkpt_save(self):
        os.makedirs(PPO_CHECKPOINT_DIR + self.town, exist_ok=True)
        path = f"{PPO_CHECKPOINT_DIR}{self.town}/ppo_latest.pth"
        torch.save(
            {
                "policy": self.old_policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "action_std": getattr(self, "action_std_init", None),
                "meta": {"checkpoint_no": getattr(self, "checkpoint_file_no", 0),
                        "saved_at": self._now(), "town": self.town, "kind": "rolling"},
            },
            path,
        )

    def save(self):
        os.makedirs(PPO_CHECKPOINT_DIR + self.town, exist_ok=True)
        n = len([f for f in next(os.walk(PPO_CHECKPOINT_DIR + self.town))[2] if f.endswith(".pth")])
        self.checkpoint_file_no = n
        path = f"{PPO_CHECKPOINT_DIR}{self.town}/ppo_policy_{n:04d}.pth"
        torch.save(
            {
                "policy": self.old_policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "action_std": getattr(self, "action_std_init", None),
                "meta": {"checkpoint_no": n, "saved_at": self._now(), "town": self.town, "kind": "final"},
            },
            path,
        )


    def load(self):
        """Load latest .pth; supports both new payload dicts and legacy raw state_dict files."""
        folder = PPO_CHECKPOINT_DIR + self.town
        files = sorted([f for f in next(os.walk(folder))[2] if f.endswith(".pth")])
        if not files:
            raise FileNotFoundError(f"No checkpoints in {folder}")
        path = os.path.join(folder, files[-1])

        blob = torch.load(path, map_location=device)
        # legacy support: file may be a raw state_dict
        if isinstance(blob, dict) and "policy" in blob:
            state_dict = blob["policy"]
            opt_state = blob.get("optimizer", None)
            meta = blob.get("meta", {})
        else:
            state_dict = blob
            opt_state = None
            meta = {}

        self.old_policy.load_state_dict(state_dict)
        self.policy.load_state_dict(state_dict)
        if opt_state is not None and hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.load_state_dict(opt_state)

        return path, meta