# FetchReach Reinforcement Learning Project

This project implements multiple reinforcement learning algorithms for the FetchReach-v1 environment from Gymnasium Robotics. The agents learn to control a robotic arm to reach a target position in 3D space. The implemented algorithms include Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), Twin Delayed DDPG (TD3), and Deep Deterministic Policy Gradient (DDPG).

## Project Structure

- `train.py`: Main training script supporting multiple algorithms (SAC, PPO, TD3, DDPG)
- `test.py`: Script to test trained models and compare different algorithms
- `visualize.py`: Script to generate comprehensive visualizations of trained agents
- `run_all_algorithms.py`: Script to train, test, and visualize all algorithms sequentially
- `run_single_algorithm.py`: Script to train, test, and visualize a single algorithm
- `requirements.txt`: Dependencies required to run the project
- `logs/`: Directory where training logs, models, and visualizations are saved
- `logs/comparisons/`: Directory for algorithm comparison results

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fetchreach-rl.git
cd fetchreach-rl
```

2. Create a conda environment and install dependencies:
```bash
conda create -n fetchreach python=3.10
conda activate fetchreach
pip install -r requirements.txt
```

## Usage

### Training

To train a new agent with any of the supported algorithms:

```bash
python train.py --algorithm SAC  # Options: SAC, PPO, TD3, DDPG
```

Additional training options:
```bash
python train.py --algorithm PPO --device cuda --seed 42
```

This will:
- Create a new log directory with a timestamp and algorithm name
- Train the agent for the specified number of epochs
- Save the model checkpoints and training logs
- Generate learning curves

### Testing

To test a trained model:

```bash
python test.py --algorithm SAC --log_dir logs/YOUR_LOG_DIRECTORY/ --model_file logs/YOUR_LOG_DIRECTORY/Tianshou_SAC_epoch1.pth
```

To compare multiple algorithms:

```bash
python test.py --compare --algorithms SAC,PPO,TD3,DDPG
```

Optional arguments:
- `--algorithm`: Algorithm to test (default: "SAC", options: "SAC", "PPO", "TD3", "DDPG")
- `--env_id`: Environment ID (default: "FetchReach-v1")
- `--seed`: Random seed (default: 0)
- `--device`: Device to run the model on (default: "cpu")
- `--num_episodes`: Number of episodes to test (default: 3)
- `--record_video`: Whether to record videos (default: True)
- `--test_random`: Whether to test random policy for comparison (default: True)
- `--compare`: Whether to compare multiple algorithms
- `--algorithms`: Comma-separated list of algorithms to compare (used with --compare)

### Visualization

To generate comprehensive visualizations of a trained agent:

```bash
python visualize.py --algorithm SAC --log_dir logs/YOUR_LOG_DIRECTORY/ --model_file logs/YOUR_LOG_DIRECTORY/Tianshou_SAC_epoch1.pth
```

Optional arguments:
- `--algorithm`: Algorithm to visualize (default: "SAC", options: "SAC", "PPO", "TD3", "DDPG")
- `--env_id`: Environment ID (default: "FetchReach-v1")
- `--seed`: Random seed (default: 0)
- `--device`: Device to run the model on (default: "cpu")
- `--num_videos`: Number of videos to generate (default: 10)
- `--num_episodes`: Number of episodes to collect data from (default: 100)

This will generate:
- Multiple video samples of the agent's performance
- Reward distribution plots
- Action distribution plots
- State space visualizations
- Success rate statistics
- An HTML page to view all visualizations

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Gymnasium 0.28+
- Gymnasium Robotics 1.2+
- Tianshou 0.5+
- Other dependencies listed in `requirements.txt`

## Results

After training, the agent learns to efficiently move the robotic arm to the target position. The performance can be evaluated through:

1. **Reward Metrics**: The average reward achieved by the agent compared to a random policy
2. **Success Rate**: The percentage of episodes where the agent successfully reaches the target
3. **Visualization**: Videos showing the agent's behavior and various plots analyzing its performance

## Running Multiple Algorithms

This project includes scripts to easily train, test, and visualize multiple algorithms:

### Running All Algorithms

To train, test, and visualize all supported algorithms (SAC, PPO, TD3, DDPG) sequentially:

```bash
python run_all_algorithms.py
```

Optional arguments:
- `--device`: Device to run the models on (default: "cpu", options: "cpu", "cuda")
- `--seed`: Random seed (default: 0)
- `--num_episodes`: Number of episodes for testing and data collection (default: 10)
- `--num_videos`: Number of videos to generate during visualization (default: 5)

This script will:
1. Train each algorithm sequentially
2. Test each trained algorithm
3. Generate visualizations for each algorithm
4. Compare all successfully trained algorithms

### Running a Single Algorithm

To train, test, and visualize a single algorithm:

```bash
python run_single_algorithm.py --algorithm SAC
```

Optional arguments:
- `--algorithm`: Algorithm to run (default: "SAC", options: "SAC", "PPO", "TD3", "DDPG")
- `--device`: Device to run the model on (default: "cpu", options: "cpu", "cuda")
- `--seed`: Random seed (default: 0)
- `--num_episodes`: Number of episodes for testing and data collection (default: 10)
- `--num_videos`: Number of videos to generate during visualization (default: 5)
- `--train`: Whether to train the model (default: True)
- `--test`: Whether to test the model (default: True)
- `--visualize`: Whether to visualize the model (default: True)
- `--log_dir`: Directory containing the trained model (optional)
- `--model_file`: Full path to the trained model file (optional)

### Comparing Different Methods

To compare the performance of different algorithms:

```bash
python test.py --compare --algorithms SAC,PPO,TD3,DDPG
```

This will:
- Run each algorithm for the specified number of episodes
- Generate comparative performance metrics
- Create visualizations showing the differences between methods

### Visualizing Comparison Results

The comparison results are visualized in various ways:

1. **Average Reward Comparison**: Bar charts showing average rewards across methods
2. **Reward Distribution Comparison**: Box plots of reward distributions for each method

All comparison visualizations are saved to the `logs/comparisons/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses the [Tianshou](https://github.com/thu-ml/tianshou) reinforcement learning library
- The environment is provided by [Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
