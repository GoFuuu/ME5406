# FetchReach Reinforcement Learning Project

This project implements a Soft Actor-Critic (SAC) reinforcement learning agent for the FetchReach-v1 environment from Gymnasium Robotics. The agent learns to control a robotic arm to reach a target position in 3D space.

## Project Structure

- `train.py`: Main training script for the SAC agent
- `test.py`: Script to test a trained model and generate videos
- `visualize.py`: Script to generate comprehensive visualizations of the trained agent
- `requirements.txt`: Dependencies required to run the project
- `logs/`: Directory where training logs, models, and visualizations are saved

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

To train a new SAC agent:

```bash
python train.py
```

This will:
- Create a new log directory with a timestamp
- Train the agent for the specified number of epochs
- Save the model checkpoints and training logs
- Generate learning curves

### Testing

To test a trained model:

```bash
python test.py --log_dir logs/YOUR_LOG_DIRECTORY/ --model_file logs/YOUR_LOG_DIRECTORY/Tianshou_SAC_epoch1.pth
```

Optional arguments:
- `--env_id`: Environment ID (default: "FetchReach-v1")
- `--seed`: Random seed (default: 0)
- `--device`: Device to run the model on (default: "cpu")
- `--num_episodes`: Number of episodes to test (default: 3)
- `--record_video`: Whether to record videos (default: True)
- `--test_random`: Whether to test random policy for comparison (default: True)

### Visualization

To generate comprehensive visualizations of a trained agent:

```bash
python visualize.py --log_dir logs/YOUR_LOG_DIRECTORY/ --model_file logs/YOUR_LOG_DIRECTORY/Tianshou_SAC_epoch1.pth
```

Optional arguments:
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses the [Tianshou](https://github.com/thu-ml/tianshou) reinforcement learning library
- The environment is provided by [Gymnasium Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)
