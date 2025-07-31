import gymnasium as gym
import numpy as np
import os
import time # Import time for potential pauses

from stable_baselines3 import PPO 
# Import your custom environment
# Make sure the file containing RocketHoopsEnv is accessible
from rocket_game_env1 import RocketHoopsEnv 

# --- Configuration ---
MODEL_PATH = "./rocket_model" # Path to your saved model file
NUM_EVAL_EPISODES = 10 # Number of episodes to run for evaluation
RENDER_DELAY = 0.01 # Small delay between frames for better visualization (in seconds)

# --- Environment Creation ---
# Create a single environment for evaluation with human rendering
# Set render_mode='human' to visualize the agent playing
eval_env = RocketHoopsEnv(render_mode="human")

# --- Load the Trained Model ---
try:
    model = PPO.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure you have trained a model and saved it to this path.")
    exit() # Exit if the model file is not found

# --- Evaluation Loop ---
print(f"\nStarting evaluation for {NUM_EVAL_EPISODES} episodes...")

total_rewards = []
hoops_passed_list = []
outcomes = [] # 'Win', 'Loss', 'Truncated'

for episode in range(NUM_EVAL_EPISODES):
    obs, info = eval_env.reset()
    done = False
    episode_reward = 0
    episode_hoops_passed = 0
    episode_outcome = 'Truncated' # Default outcome if max steps reached

    print(f"\n--- Episode {episode + 1}/{NUM_EVAL_EPISODES} ---")

    while not done:
        # Predict action from the loaded model
        # Use deterministic=True for consistent behavior during evaluation
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, terminated, truncated, info = eval_env.step(action)

        episode_reward += reward
        episode_hoops_passed = info.get('hoops_passed', episode_hoops_passed) # Update hoops passed from info

        # Check if episode is finished
        done = terminated or truncated

        # Render the environment
        eval_env.render()

        # Add a small delay
        time.sleep(RENDER_DELAY)

    # Determine episode outcome
    if terminated:
        if 'score' in info and info['score'] > 0 and info.get('hoops_passed', 0) == eval_env.NUM_HOOPS:
             # Check if it was a win by looking at score/hoops passed
             # (Assuming a successful mission has a positive score and all hoops passed)
            episode_outcome = 'Win'
        else:
            episode_outcome = 'Loss' # Crashed, flew off, etc.
    elif truncated:
        episode_outcome = 'Truncated' # Reached max steps

    total_rewards.append(episode_reward)
    hoops_passed_list.append(episode_hoops_passed)
    outcomes.append(episode_outcome)

    print(f"Episode finished. Outcome: {episode_outcome}, Total Reward: {episode_reward:.2f}, Hoops Passed: {episode_hoops_passed}")

# --- Evaluation Summary ---
print("\n--- Evaluation Summary ---")
print(f"Evaluated over {NUM_EVAL_EPISODES} episodes.")
print(f"Average Reward: {np.mean(total_rewards):.2f}")
print(f"Average Hoops Passed: {np.mean(hoops_passed_list):.2f}")
print(f"Win Rate: {outcomes.count('Win') / NUM_EVAL_EPISODES:.2f}")
print(f"Loss Rate: {outcomes.count('Loss') / NUM_EVAL_EPISODES:.2f}")
print(f"Truncation Rate: {outcomes.count('Truncated') / NUM_EVAL_EPISODES:.2f}")


# --- Clean up ---
eval_env.close()
print("\nEvaluation complete. Environment closed.")
