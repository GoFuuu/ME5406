"""
FetchReach-v1 Environment - Train, Test, and Visualize Multiple Algorithms
ME5406 Project
"""

import os
import time
import subprocess
import argparse

def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    print(f"Command: {command}")

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode != 0:
        print(f"Error running {description}. Return code: {process.returncode}")
        return False

    return True

def train_algorithm(algorithm, device, seed=0, max_epoch=None, step_per_epoch=None):
    """Train a specific algorithm"""
    command = f"python train.py --algorithm {algorithm} --device {device} --seed {seed}"

    # Add max_epoch parameter if provided
    if max_epoch is not None:
        command += f" --max_epoch {max_epoch}"

    # Add step_per_epoch parameter if provided
    if step_per_epoch is not None:
        command += f" --step_per_epoch {step_per_epoch}"

    success = run_command(command, f"Training {algorithm} algorithm")

    if not success:
        print(f"Training {algorithm} failed. Skipping testing and visualization.")
        return None

    # Find the most recent log directory for this algorithm
    log_dirs = sorted([d for d in os.listdir("logs") if d.startswith(f"Tianshou_{algorithm}_")], reverse=True)
    if not log_dirs:
        print(f"No log directories found for {algorithm}. Skipping testing and visualization.")
        return None

    log_dir = os.path.join("logs", log_dirs[0])
    model_file = os.path.join(log_dir, f"Tianshou_{algorithm}_epoch1.pth")

    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}. Skipping testing and visualization.")
        return None

    return log_dir, model_file

def test_algorithm(algorithm, log_dir, model_file, device, num_episodes=10):
    """Test a specific algorithm"""
    command = f"python test.py --algorithm {algorithm} --log_dir {log_dir} --model_file {model_file} --device {device} --num_episodes {num_episodes}"
    success = run_command(command, f"Testing {algorithm} algorithm")

    if not success:
        print(f"Testing {algorithm} failed.")
        return False

    return True

def visualize_algorithm(algorithm, log_dir, model_file, device, num_videos=5, num_episodes=50):
    """Visualize a specific algorithm"""
    command = f"python visualize.py --algorithm {algorithm} --log_dir {log_dir} --model_file {model_file} --device {device} --num_videos {num_videos} --num_episodes {num_episodes}"
    success = run_command(command, f"Visualizing {algorithm} algorithm")

    if not success:
        print(f"Visualizing {algorithm} failed.")
        return False

    return True

def compare_algorithms(algorithms, device, num_episodes=10):
    """Compare multiple algorithms"""
    algorithms_str = ",".join(algorithms)
    command = f"python test.py --compare --algorithms {algorithms_str} --device {device} --num_episodes {num_episodes}"
    success = run_command(command, f"Comparing algorithms: {algorithms_str}")

    if not success:
        print(f"Comparing algorithms failed.")
        return False

    return True

def main(args):
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create logs/comparisons directory if it doesn't exist
    os.makedirs("logs/comparisons", exist_ok=True)

    # List of algorithms to train, test, and visualize
    #algorithms = ["SAC", "PPO", "TD3", "DDPG"]
    algorithms = ["PPO", "TD3", "DDPG"]

    # Dictionary to store log directories and model files for each algorithm
    algorithm_info = {}

    # Train, test, and visualize each algorithm
    for algorithm in algorithms:
        print(f"\n\n{'#'*100}")
        print(f"Processing {algorithm} algorithm")
        print(f"{'#'*100}")

        # Train the algorithm with increased steps per epoch for better convergence
        max_epoch = args.max_epoch if args.max_epoch is not None else 1

        if algorithm == 'SAC':
            # SAC converges well with default settings
            step_per_epoch = args.step_per_epoch if args.step_per_epoch is not None else 100_000
            result = train_algorithm(algorithm, args.device, args.seed, max_epoch=max_epoch, step_per_epoch=step_per_epoch)
        elif algorithm == 'PPO':
            # PPO needs more steps to converge
            step_per_epoch = args.step_per_epoch if args.step_per_epoch is not None else 250_000
            result = train_algorithm(algorithm, args.device, args.seed, max_epoch=max_epoch, step_per_epoch=step_per_epoch)
        elif algorithm == 'TD3':
            # TD3 needs more steps to converge
            step_per_epoch = args.step_per_epoch if args.step_per_epoch is not None else 250_000
            result = train_algorithm(algorithm, args.device, args.seed, max_epoch=max_epoch, step_per_epoch=step_per_epoch)
        elif algorithm == 'DDPG':
            # DDPG needs more steps to converge
            step_per_epoch = args.step_per_epoch if args.step_per_epoch is not None else 250_000
            result = train_algorithm(algorithm, args.device, args.seed, max_epoch=max_epoch, step_per_epoch=step_per_epoch)

        if result is None:
            continue

        log_dir, model_file = result
        algorithm_info[algorithm] = (log_dir, model_file)

        # Test the algorithm
        test_algorithm(algorithm, log_dir, model_file, args.device, args.num_episodes)

        # Visualize the algorithm
        visualize_algorithm(algorithm, log_dir, model_file, args.device, args.num_videos, args.num_episodes)

    # Compare all successfully trained algorithms
    if len(algorithm_info) > 1:
        print(f"\n\n{'#'*100}")
        print(f"Comparing all algorithms")
        print(f"{'#'*100}")

        compare_algorithms(list(algorithm_info.keys()), args.device, args.num_episodes)

    print("\n\nAll processing completed!")
    print(f"Successfully processed algorithms: {list(algorithm_info.keys())}")

    # Print paths to results
    print("\nResults can be found in the following directories:")
    for algorithm, (log_dir, _) in algorithm_info.items():
        print(f"{algorithm}: {log_dir}")

    if len(algorithm_info) > 1:
        print(f"Comparisons: logs/comparisons")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, test, and visualize multiple reinforcement learning algorithms")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run the models on (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes for testing and data collection")
    parser.add_argument("--num_videos", type=int, default=5,
                        help="Number of videos to generate during visualization")
    parser.add_argument("--max_epoch", type=int, default=None,
                        help="Maximum number of epochs to train (default: 1)")
    parser.add_argument("--step_per_epoch", type=int, default=None,
                        help="Number of steps per epoch (default: varies by algorithm)")

    args = parser.parse_args()
    main(args)
