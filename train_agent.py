import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# Import your custom environment
from rocket_game_env1 import RocketHoopsEnv 

# --- Configuration ---
LOG_DIR = "./rocket_log/"
MODEL_SAVE_PATH = "./rocket_model"
TOTAL_TIMESTEPS = 500000 # Number of timesteps to train the agent
NUM_ENVS = 4 # Number of parallel environments for training
SEED = 42 # Random seed for reproducibility

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# --- Environment Creation ---

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # Pass render_mode=None during training for speed
        env = RocketHoopsEnv(render_mode=None)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    # Create vectorized environment for parallel training
    # Use SubprocVecEnv for potentially faster training with multiple cores
    vec_env = SubprocVecEnv([make_env("RocketHoopsEnv", i, seed=SEED) for i in range(NUM_ENVS)])

    # --- Model Definition ---
    # Define the PPO model
    # 'MlpPolicy' is a standard feedforward neural network policy
    # verbose=1 logs training progress
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)
    # model = PPO(
    # "MlpPolicy",
    # vec_env,
    # verbose=1,
    # tensorboard_log=LOG_DIR,
    # # Hyperparameter adjustments
    # learning_rate=3e-4,                # Consider using a schedule: linear_schedule(3e-4)
    # n_steps=2048,                       # Number of steps per update
    # batch_size=64,                      # Minibatch size
    # n_epochs=10,                       # Number of optimization epochs per update
    # gamma=0.99,                        # Discount factor
    # gae_lambda=0.95,                   # GAE parameter
    # clip_range=0.2,                    # Initial clip range (can be schedule)
    # clip_range_vf=None,                # Separate clip range for value function
    # ent_coef=0.01,                     # Entropy regularization coefficient
    # vf_coef=0.5,                       # Value function loss coefficient
    # max_grad_norm=0.5,                 # Gradient clipping
    # use_sde=True,                      # Use generalized State Dependent Exploration
    # sde_sample_freq=64,                # SDE sample frequency
    # policy_kwargs={
    #     "net_arch": [dict(pi=[256, 256], vf=[256, 256])],  # Larger network
    #     "activation_fn": torch.nn.Tanh,
    #     "ortho_init": True,
    #     "log_std_init": -0.5
    # }
    # )

    # --- Training ---
    print(f"Training started for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    print("Training finished.")

    # --- Saving the Model ---
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
