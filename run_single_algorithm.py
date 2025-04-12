"""
FetchReach-v1 Environment - Train, Test, and Visualize a Single Algorithm
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

def main(args):
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    algorithm = args.algorithm
    
    # Step 1: Training
    if args.train:
        command = f"python train.py --algorithm {algorithm} --device {args.device} --seed {args.seed}"
        success = run_command(command, f"Training {algorithm} algorithm")
        
        if not success:
            print(f"Training {algorithm} failed. Exiting.")
            return
    
    # Find the most recent log directory for this algorithm if not specified
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dirs = sorted([d for d in os.listdir("logs") if d.startswith(f"Tianshou_{algorithm}_")], reverse=True)
        if not log_dirs:
            print(f"No log directories found for {algorithm}. Please train the model first or specify a log directory.")
            return
        log_dir = os.path.join("logs", log_dirs[0])
    
    # Find the model file if not specified
    if args.model_file:
        model_file = args.model_file
    else:
        model_file = os.path.join(log_dir, f"Tianshou_{algorithm}_epoch1.pth")
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}. Please train the model first or specify a valid model file.")
        return
    
    # Step 2: Testing
    if args.test:
        command = f"python test.py --algorithm {algorithm} --log_dir {log_dir} --model_file {model_file} --device {args.device} --num_episodes {args.num_episodes}"
        success = run_command(command, f"Testing {algorithm} algorithm")
        
        if not success:
            print(f"Testing {algorithm} failed. Skipping visualization.")
            return
    
    # Step 3: Visualization
    if args.visualize:
        command = f"python visualize.py --algorithm {algorithm} --log_dir {log_dir} --model_file {model_file} --device {args.device} --num_videos {args.num_videos} --num_episodes {args.num_episodes}"
        success = run_command(command, f"Visualizing {algorithm} algorithm")
        
        if not success:
            print(f"Visualizing {algorithm} failed.")
            return
    
    print(f"\nAll processing for {algorithm} completed!")
    print(f"Results can be found in: {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, test, and visualize a single reinforcement learning algorithm")
    parser.add_argument("--algorithm", type=str, default="SAC", choices=["SAC", "PPO", "TD3", "DDPG"],
                        help="RL algorithm to use")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run the model on (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes for testing and data collection")
    parser.add_argument("--num_videos", type=int, default=5,
                        help="Number of videos to generate during visualization")
    parser.add_argument("--log_dir", type=str, default="",
                        help="Directory containing the trained model (optional)")
    parser.add_argument("--model_file", type=str, default="",
                        help="Full path to the trained model file (optional)")
    parser.add_argument("--train", action="store_true", default=True,
                        help="Whether to train the model")
    parser.add_argument("--test", action="store_true", default=True,
                        help="Whether to test the model")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Whether to visualize the model")
    
    args = parser.parse_args()
    main(args)
