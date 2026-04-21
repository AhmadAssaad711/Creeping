"""

    All the much needed hyper-parameters needed for the algorithm implementation. 

"""

MODEL_LOAD = False
SEED = 0
BATCH_SIZE = 1
IM_WIDTH = 160
IM_HEIGHT = 80
GAMMA = 0.99
MEMORY_SIZE = 5000
EPISODES = 1000

# Stage-1 shapes (YOU: keep these consistent with your trained Stage-1)
OBS_OLD_DIM = 5            # [throttle, velocity, prev_steer, distance_from_center, angle]
LATENT_DIM_S1 = 95         # so 95 + 5 = 100
STAGE1_OBS_DIM = 100       # must match checkpoint

# Paths for Stage-1 components
S1_ACTOR_WEIGHTS   = "checkpoints_stage1/stage1_actor.pt"

#VAE Bottleneck
LATENT_DIM = 95

#Proximal Policy Optimization (hyper)parameters
EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 2e6
ACTION_STD_INIT = 0.2
TEST_TIMESTEPS = 5e4
PPO_LEARNING_RATE = 1e-4  
PPO_CHECKPOINT_DIR = 'checkpoints/'
PPO_FINAL_DIR = 'preTrained_models/'
POLICY_CLIP = 0.2

# Testing
TEST_TIMESTEPS = 1000