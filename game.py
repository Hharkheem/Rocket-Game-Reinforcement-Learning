import gymnasium as gym
import pygame
import numpy as np
import csv
import os
from stable_baselines3 import PPO, DQN
from rocket_game_env1 import RocketHoopsEnv

# === Config ===
MODEL_PATH = "rocket_model"
LOG_FILE = "scores.csv"
EPISODES = 100  # How many episodes to play

# === Load model ===
model = PPO.load(MODEL_PATH)

# === Setup logging ===
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Score'])

# === Environment ===
env = RocketHoopsEnv(render_mode="human")
episode_count = 0
scores = []

for episode in range(EPISODES):
    obs, _ = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        score += reward

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

    # === Score logging ===
    episode_count += 1
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([episode_count, round(score, 2)])
    scores.append(score)
    print(f"Episode {episode_count} | Score: {round(score, 2)}")

env.close()
