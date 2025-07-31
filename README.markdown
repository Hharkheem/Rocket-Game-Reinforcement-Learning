# Rocket Hoops Reinforcement Learning Project

## Overview

This repository contains a reinforcement learning (RL) project where a rocket navigates through a series of hoops in a 2D environment. The project uses a custom Gymnasium environment (`RocketHoopsEnv`) and trains an agent using the PPO algorithm from Stable Baselines3. The goal is for the rocket to pass through hoops while avoiding obstacles, with rewards based on speed and success.

## Features

- **Custom Environment**: A 2D rocket navigation game built with Pygame and Gymnasium.
- **RL Training**: Uses PPO to train the agent with parallel environments for efficiency.
- **Evaluation**: Script to evaluate the trained model with visualization and performance metrics.
- **Game Mode**: Play the game with the trained model and log scores to a CSV file.
- **Dynamic Hoops**: Hoops move with increasing speed, and "death traps" add challenge after passing regular hoops.
- **Reward System**: Time-based rewards for passing hoops and a completion bonus for finishing all hoops.

## Repository Structure

- `rocket_game_env1.py`: Defines the `RocketHoopsEnv` custom environment, including rocket physics, hoop mechanics, and rendering.
- `train_agent.py`: Trains the PPO model using parallel environments and saves the trained model.
- `evaluate_agent.py`: Evaluates the trained model over multiple episodes, displaying performance metrics and visualizing gameplay.
- `game.py`: Runs the game with the trained model, logs scores to a CSV file, and renders the gameplay.
- `rocket_model/`: Directory where the trained PPO model is saved (generated after training).
- `rocket_log/`: Directory for TensorBoard logs (generated during training).
- `scores.csv`: CSV file for logging scores from `game.py` (generated during gameplay).

## Requirements

- Python 3.8+
- Libraries:
  ```bash
  pip install gymnasium pygame stable-baselines3 torch numpy
  ```
- Optional: TensorBoard for visualizing training logs:
  ```bash
  pip install tensorboard
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hharkheem/Rocket-Game-Reinforcement-Learning
   cd Rocket-Game-Reinforcement-Learning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   (Create a `requirements.txt` with the above libraries if desired.)
3. Ensure Pygame is properly installed for rendering.

## Usage

### 1. Train the Agent

Run the training script to train the PPO model:

```bash
python train_agent.py
```

- Trains for 500,000 timesteps using 4 parallel environments.
- Saves the model to `rocket_model`.
- Logs training progress to `rocket_log/` (view with TensorBoard: `tensorboard --logdir rocket_log/`).

### 2. Evaluate the Trained Model

Evaluate the trained model with visualization:

```bash
python evaluate_agent.py
```

- Runs 10 episodes, rendering the gameplay.
- Outputs average reward, hoops passed, win/loss/truncation rates.
- Requires a trained model at `rocket_model`.

### 3. Play the Game

Run the game with the trained model and log scores:

```bash
python game.py
```

- Plays 100 episodes, rendering the gameplay.
- Logs scores to `scores.csv`.
- Requires a trained model at `rocket_model`.

### 4. View Training Logs

To visualize training progress:

```bash
tensorboard --logdir rocket_log/
```

Open the provided URL in a browser.

## Environment Details

- **Action Space**: Discrete (5 actions: No-op, Up, Down, Left, Right).
- **Observation Space**: Continuous (6D vector: rocket x/y position, x/y velocity, x/y distance to next hoop).
- **Rewards**:
  - Small per-frame penalty (-0.01) to encourage speed.
  - Time-based reward for passing hoops (2.0 to 5.0 based on speed).
  - Large time-based completion bonus (100.0+) for passing all 5 hoops.
  - Penalty (-10) for hitting a death trap.
- **Termination**: Episode ends on hitting a death trap, completing all hoops, or manual exit.
- **Rendering**: Supports "human" (Pygame window) and "rgb_array" modes.

## Notes

- The commented-out hyperparameters in `train_agent.py` provide options for tuning the PPO model (e.g., learning rate, network architecture).
- Adjust `NUM_HOOPS`, `BASE_HOOP_SPEED`, or `SPEED_INCREMENT` in `rocket_game_env1.py` to change difficulty.
- Ensure the trained model (`rocket_model`) exists before running `evaluate_agent.py` or `game.py`.
- The environment uses Pygame for rendering, which may require a display server (e.g., X11 on Linux).

## Future Improvements

- Add more complex obstacles or hoop patterns.
- Experiment with other RL algorithms (e.g., DQN, SAC).
- Implement curriculum learning to gradually increase difficulty.
- Optimize rendering for smoother visualization on low-end systems.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/), [Pygame](https://www.pygame.org/), and [Stable Baselines3](https://stable-baselines3.readthedocs.io/).
- Inspired by RL applications in game environments.
