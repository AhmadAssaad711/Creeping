"""
parameters.py
-------------
Hyperparameters for PPO + SUMO creeping negotiation.
"""

MODEL_LOAD = False
SEED = 0
GAMMA = 0.99

# PPO
PPO_LEARNING_RATE  = 3e-4        # was 1e-4, too slow to converge
ACTION_STD_INIT    = 0.4         # initial exploration std
POLICY_CLIP        = 0.2
EPISODE_LENGTH     = 500         # was 7500 -- way too long; PPO updates every 10 eps
PPO_CHECKPOINT_DIR = 'ppo_checkpoints/'
PPO_FINAL_DIR      = 'ppo_checkpoints/'